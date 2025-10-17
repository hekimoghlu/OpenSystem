/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef MODULES_RTP_RTCP_SOURCE_FORWARD_ERROR_CORRECTION_INTERNAL_H_
#define MODULES_RTP_RTCP_SOURCE_FORWARD_ERROR_CORRECTION_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>

#include "api/array_view.h"
#include "modules/include/module_fec_types.h"

namespace webrtc {

// Maximum number of media packets that can be protected
// by these packet masks.
constexpr size_t kUlpfecMaxMediaPackets = 48;

// Packet mask size in bytes (given L bit).
constexpr size_t kUlpfecPacketMaskSizeLBitClear = 2;
constexpr size_t kUlpfecPacketMaskSizeLBitSet = 6;

// Packet code mask maximum length. kFECPacketMaskMaxSize = MaxNumFECPackets *
// (kUlpfecMaxMediaPackets / 8), and MaxNumFECPackets is equal to maximum number
// of media packets (kUlpfecMaxMediaPackets)
constexpr size_t kFECPacketMaskMaxSize = 288;

// Convenience constants.
constexpr size_t kUlpfecMinPacketMaskSize = kUlpfecPacketMaskSizeLBitClear;
constexpr size_t kUlpfecMaxPacketMaskSize = kUlpfecPacketMaskSizeLBitSet;

namespace internal {

class PacketMaskTable {
 public:
  PacketMaskTable(FecMaskType fec_mask_type, int num_media_packets);
  ~PacketMaskTable();

  rtc::ArrayView<const uint8_t> LookUp(int num_media_packets,
                                       int num_fec_packets);

 private:
  static const uint8_t* PickTable(FecMaskType fec_mask_type,
                                  int num_media_packets);
  const uint8_t* table_;
  uint8_t fec_packet_mask_[kFECPacketMaskMaxSize];
};

rtc::ArrayView<const uint8_t> LookUpInFecTable(const uint8_t* table,
                                               int media_packet_index,
                                               int fec_index);

// Returns an array of packet masks. The mask of a single FEC packet
// corresponds to a number of mask bytes. The mask indicates which
// media packets should be protected by the FEC packet.

// \param[in]  num_media_packets       The number of media packets to protect.
//                                     [1, max_media_packets].
// \param[in]  num_fec_packets         The number of FEC packets which will
//                                     be generated. [1, num_media_packets].
// \param[in]  num_imp_packets         The number of important packets.
//                                     [0, num_media_packets].
//                                     num_imp_packets = 0 is the equal
//                                     protection scenario.
// \param[in]  use_unequal_protection  Enables unequal protection: allocates
//                                     more protection to the num_imp_packets.
// \param[in]  mask_table              An instance of the `PacketMaskTable`
//                                     class, which contains the type of FEC
//                                     packet mask used, and a pointer to the
//                                     corresponding packet masks.
// \param[out] packet_mask             A pointer to hold the packet mask array,
//                                     of size: num_fec_packets *
//                                     "number of mask bytes".
void GeneratePacketMasks(int num_media_packets,
                         int num_fec_packets,
                         int num_imp_packets,
                         bool use_unequal_protection,
                         PacketMaskTable* mask_table,
                         uint8_t* packet_mask);

// Returns the required packet mask size, given the number of sequence numbers
// that will be covered.
size_t PacketMaskSize(size_t num_sequence_numbers);

// Inserts `num_zeros` zero columns into `new_mask` at position
// `new_bit_index`. If the current byte of `new_mask` can't fit all zeros, the
// byte will be filled with zeros from `new_bit_index`, but the next byte will
// be untouched.
void InsertZeroColumns(int num_zeros,
                       uint8_t* new_mask,
                       int new_mask_bytes,
                       int num_fec_packets,
                       int new_bit_index);

// Copies the left most bit column from the byte pointed to by
// `old_bit_index` in `old_mask` to the right most column of the byte pointed
// to by `new_bit_index` in `new_mask`. `old_mask_bytes` and `new_mask_bytes`
// represent the number of bytes used per row for each mask. `num_fec_packets`
// represent the number of rows of the masks.
// The copied bit is shifted out from `old_mask` and is shifted one step to
// the left in `new_mask`. `new_mask` will contain "xxxx xxn0" after this
// operation, where x are previously inserted bits and n is the new bit.
void CopyColumn(uint8_t* new_mask,
                int new_mask_bytes,
                uint8_t* old_mask,
                int old_mask_bytes,
                int num_fec_packets,
                int new_bit_index,
                int old_bit_index);

}  // namespace internal
}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_FORWARD_ERROR_CORRECTION_INTERNAL_H_
