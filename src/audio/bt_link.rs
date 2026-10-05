//! Bluetooth voice-link (SCO/eSCO) state, read from the kernel.
//!
//! A Bluetooth headset microphone in HFP mode only carries audio once
//! the synchronous voice link (SCO or eSCO) to the headset is up.  Until
//! then PipeWire's `bluez_input.*` node streams exact zeros — and so
//! does a live headset whose noise gate is closed, so sample values can
//! never tell the two apart.  The kernel's HCI connection list can: it
//! is the same `HCIGETCONNLIST` ioctl `hcitool con` uses, and needs no
//! privileges.

use crate::error::TalkError;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

/// `_IOR('H', 210, int)`: list HCI controllers.
const HCIGETDEVLIST: libc::c_ulong = 0x8004_48D2;
/// `_IOR('H', 212, int)`: list the connections of one controller.
const HCIGETCONNLIST: libc::c_ulong = 0x8004_48D4;
/// `BTPROTO_HCI` socket protocol.
const BTPROTO_HCI: libc::c_int = 1;
/// Upper bound of controllers / connections requested per ioctl.
const MAX_ENTRIES: usize = 16;
/// `struct hci_dev_req` size (u16 dev_id + padding + u32 dev_opt).
const DEV_REQ_SIZE: usize = 8;
/// `struct hci_conn_info` size (u16 handle, bdaddr_t, u8 type, u8 out,
/// u16 state, u32 link_mode).
const CONN_INFO_SIZE: usize = 16;
/// Header of the list requests (u16 count fields, padded to 4).
const LIST_HEADER_SIZE: usize = 4;
/// HCI link types carrying voice.
const SCO_LINK: u8 = 0x00;
const ESCO_LINK: u8 = 0x02;
/// `BT_CONNECTED` connection state.
const BT_CONNECTED: u16 = 1;

/// One entry of a controller's connection list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConnInfo {
    /// Remote address, in kernel (little-endian) byte order.
    bdaddr: [u8; 6],
    link_type: u8,
    state: u16,
}

/// Bluetooth address of a PipeWire HFP source named
/// `bluez_input.AA_BB_CC_DD_EE_FF[.N]`, in kernel byte order.
///
/// `None` for any other source (not a Bluetooth microphone).
pub(crate) fn bt_address_of_source(source: &str) -> Option<[u8; 6]> {
    let rest = source.strip_prefix("bluez_input.")?;
    let address = rest.split('.').next()?;
    let octets: Vec<u8> = address
        .split(['_', ':'])
        .map(|octet| u8::from_str_radix(octet, 16))
        .collect::<Result<_, _>>()
        .ok()?;
    let octets: [u8; 6] = octets.try_into().ok()?;
    let mut kernel_order = octets;
    kernel_order.reverse();
    Some(kernel_order)
}

/// Whether a connected voice link (SCO or eSCO) to `bdaddr` exists.
fn has_voice_link(connections: &[ConnInfo], bdaddr: [u8; 6]) -> bool {
    connections.iter().any(|conn| {
        conn.bdaddr == bdaddr
            && matches!(conn.link_type, SCO_LINK | ESCO_LINK)
            && conn.state == BT_CONNECTED
    })
}

/// Parse a `struct hci_conn_list_req` buffer filled by the kernel.
fn parse_conn_list(buffer: &[u8]) -> Vec<ConnInfo> {
    let count = usize::from(u16::from_le_bytes([buffer[2], buffer[3]])).min(MAX_ENTRIES);
    (0..count)
        .filter_map(|index| {
            let start = LIST_HEADER_SIZE + index * CONN_INFO_SIZE;
            let entry = buffer.get(start..start + CONN_INFO_SIZE)?;
            let mut bdaddr = [0u8; 6];
            bdaddr.copy_from_slice(&entry[2..8]);
            Some(ConnInfo {
                bdaddr,
                link_type: entry[8],
                state: u16::from_le_bytes([entry[10], entry[11]]),
            })
        })
        .collect()
}

fn hci_socket() -> Result<OwnedFd, TalkError> {
    // SAFETY: plain socket(2) call; the descriptor is owned right away.
    let fd = unsafe { libc::socket(libc::AF_BLUETOOTH, libc::SOCK_RAW, BTPROTO_HCI) };
    if fd < 0 {
        return Err(TalkError::Audio(format!(
            "cannot open Bluetooth HCI socket: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: `fd` is a freshly created, valid descriptor we own.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn ioctl_list(fd: &OwnedFd, request: libc::c_ulong, buffer: &mut [u8]) -> Result<(), TalkError> {
    // SAFETY: `buffer` is sized for the header plus MAX_ENTRIES entries,
    // matching the count announced in its header.
    let result = unsafe { libc::ioctl(fd.as_raw_fd(), request, buffer.as_mut_ptr()) };
    if result < 0 {
        return Err(TalkError::Audio(format!(
            "Bluetooth HCI ioctl failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

/// Every connection of every local Bluetooth controller.
fn connections() -> Result<Vec<ConnInfo>, TalkError> {
    let fd = hci_socket()?;
    let mut devices = vec![0u8; LIST_HEADER_SIZE + MAX_ENTRIES * DEV_REQ_SIZE];
    devices[..2].copy_from_slice(&(MAX_ENTRIES as u16).to_le_bytes());
    ioctl_list(&fd, HCIGETDEVLIST, &mut devices)?;
    let device_count = usize::from(u16::from_le_bytes([devices[0], devices[1]])).min(MAX_ENTRIES);

    let mut all = Vec::new();
    for index in 0..device_count {
        let start = LIST_HEADER_SIZE + index * DEV_REQ_SIZE;
        let dev_id = [devices[start], devices[start + 1]];
        let mut list = vec![0u8; LIST_HEADER_SIZE + MAX_ENTRIES * CONN_INFO_SIZE];
        list[..2].copy_from_slice(&dev_id);
        list[2..4].copy_from_slice(&(MAX_ENTRIES as u16).to_le_bytes());
        ioctl_list(&fd, HCIGETCONNLIST, &mut list)?;
        all.extend(parse_conn_list(&list));
    }
    Ok(all)
}

/// Check whether the voice link to the headset owning `source` is up.
pub(crate) type LinkCheck = Box<dyn FnMut() -> Result<bool, TalkError> + Send>;

/// For a Bluetooth HFP source, a check of its voice link; `None` for
/// any other source (whose first buffer is proof enough).
pub(crate) fn voice_link_check(source: &str) -> Option<LinkCheck> {
    let bdaddr = bt_address_of_source(source)?;
    Some(Box::new(move || {
        Ok(has_voice_link(&connections()?, bdaddr))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_of_bluez_input_is_parsed_in_kernel_byte_order() {
        assert_eq!(
            bt_address_of_source("bluez_input.7C_E9_13_93_6C_E6.0"),
            Some([0xE6, 0x6C, 0x93, 0x13, 0xE9, 0x7C])
        );
        assert_eq!(
            bt_address_of_source("bluez_input.7C:E9:13:93:6C:E6"),
            Some([0xE6, 0x6C, 0x93, 0x13, 0xE9, 0x7C])
        );
    }

    #[test]
    fn non_bluetooth_sources_have_no_address() {
        assert_eq!(
            bt_address_of_source("alsa_input.pci-0000_c1_00.6.HiFi__hw_Generic_1__source"),
            None
        );
        assert_eq!(bt_address_of_source("bluez_input.not_an_address.0"), None);
    }

    fn conn(bdaddr: [u8; 6], link_type: u8, state: u16) -> ConnInfo {
        ConnInfo {
            bdaddr,
            link_type,
            state,
        }
    }

    #[test]
    fn only_a_connected_voice_link_to_that_headset_counts() {
        let headset = [0xE6, 0x6C, 0x93, 0x13, 0xE9, 0x7C];
        let other = [1, 2, 3, 4, 5, 6];
        // ACL (data) link only: profile switched, voice link not up yet.
        assert!(!has_voice_link(
            &[conn(headset, 0x01, BT_CONNECTED)],
            headset
        ));
        // Voice link still connecting.
        assert!(!has_voice_link(&[conn(headset, ESCO_LINK, 2)], headset));
        // Another device's voice link.
        assert!(!has_voice_link(
            &[conn(other, ESCO_LINK, BT_CONNECTED)],
            headset
        ));
        assert!(has_voice_link(
            &[
                conn(headset, 0x01, BT_CONNECTED),
                conn(headset, ESCO_LINK, BT_CONNECTED)
            ],
            headset
        ));
        assert!(has_voice_link(
            &[conn(headset, SCO_LINK, BT_CONNECTED)],
            headset
        ));
    }

    #[test]
    fn kernel_conn_list_layout_is_parsed() {
        let mut buffer = vec![0u8; LIST_HEADER_SIZE + 2 * CONN_INFO_SIZE];
        buffer[2..4].copy_from_slice(&2u16.to_le_bytes());
        // Entry 0: ACL handle 50.
        buffer[4..6].copy_from_slice(&50u16.to_le_bytes());
        buffer[6..12].copy_from_slice(&[0xE6, 0x6C, 0x93, 0x13, 0xE9, 0x7C]);
        buffer[12] = 0x01;
        buffer[14..16].copy_from_slice(&1u16.to_le_bytes());
        // Entry 1: eSCO handle 3584.
        buffer[20..22].copy_from_slice(&3584u16.to_le_bytes());
        buffer[22..28].copy_from_slice(&[0xE6, 0x6C, 0x93, 0x13, 0xE9, 0x7C]);
        buffer[28] = ESCO_LINK;
        buffer[30..32].copy_from_slice(&1u16.to_le_bytes());
        let parsed = parse_conn_list(&buffer);
        assert_eq!(
            parsed,
            vec![
                conn([0xE6, 0x6C, 0x93, 0x13, 0xE9, 0x7C], 0x01, 1),
                conn([0xE6, 0x6C, 0x93, 0x13, 0xE9, 0x7C], ESCO_LINK, 1),
            ]
        );
    }
}
