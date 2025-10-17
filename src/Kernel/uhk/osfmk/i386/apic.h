/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
/*
 * @OSF_COPYRIGHT@
 *
 */
#ifndef _I386_APIC_H_
#define _I386_APIC_H_

#define IOAPIC_START                    0xFEC00000
#define IOAPIC_SIZE                     0x00000020

#define IOAPIC_RSELECT                  0x00000000
#define IOAPIC_RWINDOW                  0x00000010
#define IOA_R_ID                        0x00
#define         IOA_R_ID_SHIFT          24
#define IOA_R_VERSION                   0x01
#define         IOA_R_VERSION_MASK      0xFF
#define         IOA_R_VERSION_ME_SHIFT  16
#define         IOA_R_VERSION_ME_MASK   0xFF
#define IOA_R_REDIRECTION               0x10
#define         IOA_R_R_VECTOR_MASK     0x000FF
#define         IOA_R_R_DM_MASK         0x00700
#define         IOA_R_R_DM_FIXED        0x00000
#define         IOA_R_R_DM_LOWEST       0x00100
#define         IOA_R_R_DM_NMI          0x00400
#define         IOA_R_R_DM_RESET        0x00500
#define         IOA_R_R_DM_EXTINT       0x00700
#define         IOA_R_R_DEST_LOGICAL    0x00800
#define         IOA_R_R_DS_PENDING      0x01000
#define         IOA_R_R_IP_PLRITY_LOW   0x02000
#define         IOA_R_R_TM_LEVEL        0x08000
#define         IOA_R_R_MASKED          0x10000

#endif /* _I386_APIC_H_ */
