/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
#ifndef _UAPI_LINUX_TOSHIBA_H
#define _UAPI_LINUX_TOSHIBA_H
#define TOSH_PROC "/proc/toshiba"
#define TOSH_DEVICE "/dev/toshiba"
#define TOSHIBA_ACPI_PROC "/proc/acpi/toshiba"
#define TOSHIBA_ACPI_DEVICE "/dev/toshiba_acpi"
typedef struct {
  unsigned int eax;
  unsigned int ebx __attribute__((packed));
  unsigned int ecx __attribute__((packed));
  unsigned int edx __attribute__((packed));
  unsigned int esi __attribute__((packed));
  unsigned int edi __attribute__((packed));
} SMMRegisters;
#define TOSH_SMM _IOWR('t', 0x90, SMMRegisters)
#define TOSHIBA_ACPI_SCI _IOWR('t', 0x91, SMMRegisters)
#endif
