/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
 * Copyright (c) 2024 Mark Kettenis <kettenis@openbsd.org>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <sys/param.h>

#include <efi.h>
#include <efiapi.h>

#define EFI_DT_FIXUP_PROTOCOL_GUID		\
  { 0xe617d64c, 0xfe08, 0x46da, \
    { 0xf4, 0xdc, 0xbb, 0xd5, 0x87, 0x0c, 0x73, 0x00 } }

INTERFACE_DECL(_EFI_DT_FIXUP_PROTOCOL);

typedef EFI_STATUS
(EFIAPI *EFI_DT_FIXUP) (
    IN struct _EFI_DT_FIXUP_PROTOCOL	*This,
    IN VOID				*Fdt,
    IN OUT UINTN			*BufferSize,
    IN UINT32				Flags
    );

#define EFI_DT_APPLY_FIXUPS	0x00000001
#define EFI_DT_RESERVE_MEMORY	0x00000002

typedef struct _EFI_DT_FIXUP_PROTOCOL {
	UINT64			Revision;
	EFI_DT_FIXUP		Fixup;
} EFI_DT_FIXUP_PROTOCOL;
