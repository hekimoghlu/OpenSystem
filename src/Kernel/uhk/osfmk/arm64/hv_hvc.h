/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#pragma once

/* Function Identifiers. */
#define HVC_FID_FAST_CALL    0x80000000
#define HVC_FID_HVC64        0x40000000
#define HVC_FID_CPU          0x01000000
#define HVC_FID_OEM          0x03000000

#define HVC_FID_SC_MASK      0xff000000
#define HVC_FID_NUM_MASK     0x0000ffff

#define HVC_FID_UID          0xff01
#define HVC_FID_REVISION     0xff03
#define HVC_FID_FEATURES     0xfeff

#define HVC32_FI(rng, num) (HVC_FID_FAST_CALL | (rng) | (num))

/* CPU */
#define HVC_CPU_SERVICE    (HVC_FID_FAST_CALL | HVC_FID_HVC64 | HVC_FID_CPU)
#define HVC32_CPU_SERVICE  (HVC_FID_FAST_CALL | HVC_FID_CPU)

/* Apple CPU Service */
#define VMAPPLE_PAC_SET_INITIAL_STATE          (HVC_CPU_SERVICE | 0x0)
#define VMAPPLE_PAC_GET_DEFAULT_KEYS           (HVC_CPU_SERVICE | 0x1)
#define VMAPPLE_PAC_SET_A_KEYS                 (HVC_CPU_SERVICE | 0x2)
#define VMAPPLE_PAC_SET_B_KEYS                 (HVC_CPU_SERVICE | 0x3)
#define VMAPPLE_PAC_SET_EL0_DIVERSIFIER        (HVC_CPU_SERVICE | 0x4)
#define VMAPPLE_PAC_SET_EL0_DIVERSIFIER_AT_EL1 (HVC_CPU_SERVICE | 0x5)
#define VMAPPLE_PAC_SET_G_KEY                  (HVC_CPU_SERVICE | 0x6)
#define VMAPPLE_PAC_NOP                        (HVC_CPU_SERVICE | 0xf0)

/* OEM */
#define HVC_OEM_SERVICE    (HVC_FID_FAST_CALL | HVC_FID_HVC64 | HVC_FID_OEM)
#define HVC32_OEM_SERVICE  (HVC_FID_FAST_CALL | HVC_FID_OEM)

/* Apple OEM Service */
#define VMAPPLE_GET_MABS_OFFSET                (HVC_OEM_SERVICE | 0x3)
#define VMAPPLE_GET_BOOTSESSIONUUID            (HVC_OEM_SERVICE | 0x4)
#define VMAPPLE_VCPU_WFK                       (HVC_OEM_SERVICE | 0x5)
#define VMAPPLE_VCPU_KICK                      (HVC_OEM_SERVICE | 0x6)

/* Apple OEM Version 1.0 */
#define HVC32_OEM_MAJOR_VER 1
#define HVC32_OEM_MINOR_VER 0

/* Common UUID identifying Apple as the implementor. */
#define VMAPPLE_HVC_UID "3B878185-AA62-4E1F-9DC9-D6799CBB6EBB"
