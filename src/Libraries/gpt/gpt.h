/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#ifndef _GPT_H_
#define	_GPT_H_

#ifdef __APPLE__
#include <libkern/OSByteOrder.h>
#include <IOKit/storage/IOGUIDPartitionScheme.h>
#else
#include <sys/endian.h>
#include <sys/gpt.h>
#endif

#ifdef __APPLE__
#include <uuid/uuid.h>
#else
#include <uuid.h>
#endif

#ifdef __APPLE__
#ifndef htole16
#define htole16(x)  OSSwapHostToLittleInt16((x))
#endif
#ifndef htole32
#define htole32(x)  OSSwapHostToLittleInt32((x))
#endif
#ifndef htole64
#define htole64(x)  OSSwapHostToLittleInt64((x))
#endif
#ifndef le16toh
#define le16toh(x)  OSSwapLittleToHostInt16((x))
#endif
#ifndef le32toh
#define le32toh(x)  OSSwapLittleToHostInt32((x))
#endif
#ifndef le64toh
#define le64toh(x)  OSSwapLittleToHostInt64((x))
#endif
#ifndef uuid_create
#define uuid_create(x, z)  uuid_generate(*(x))
#endif
#ifndef uuid_create_nil
#define uuid_create_nil(x, z)  uuid_copy(*(x), GPT_ENT_TYPE_UNUSED)
#endif
#ifndef uuid_equal
#define uuid_equal(x, y, z)  !uuid_compare(*(x), *(y))
#endif
#ifndef uuid_is_nil
#define uuid_is_nil(x, z)  uuid_is_null(*(x))
#endif
#ifndef uuid_to_string
#define uuid_to_string(x, y, z)  *(y) = malloc(40);  uuid_unparse(*(x), *(y))
#endif
UUID_DEFINE(GPT_ENT_TYPE_APPLE_HFS,0x48,0x46,0x53,0x00,0x00,0x00,0x11,0xAA,0xAA,0x11,0x00,0x30,0x65,0x43,0xEC,0xAC);
UUID_DEFINE(GPT_ENT_TYPE_APPLE_APFS,0x7C,0x34,0x57,0xEF,0x00,0x00,0x11,0xAA,0xAA,0x11,0x00,0x30,0x65,0x43,0xEC,0xAC);
UUID_DEFINE(GPT_ENT_TYPE_EFI,0xC1,0x2A,0x73,0x28,0xF8,0x1F,0x11,0xD2,0xBA,0x4B,0x00,0xA0,0xC9,0x3E,0xC9,0x3B);
UUID_DEFINE(GPT_ENT_TYPE_MS_BASIC_DATA,0xEB,0xD0,0xA0,0xA2,0xB9,0xE5,0x44,0x33,0x87,0xC0,0x68,0xB6,0xB7,0x26,0x99,0xC7);
UUID_DEFINE(GPT_ENT_TYPE_UNUSED,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00);
#endif

void	le_uuid_dec(void const *, uuid_t *);
void	le_uuid_enc(void *, uuid_t const *);
int	parse_uuid(const char *, uuid_t *);

struct mbr_part {
	uint8_t		part_flag;		/* bootstrap flags */
	uint8_t		part_shd;		/* starting head */
	uint8_t		part_ssect;		/* starting sector */
	uint8_t		part_scyl;		/* starting cylinder */
	uint8_t		part_typ;		/* partition type */
	uint8_t		part_ehd;		/* end head */
	uint8_t		part_esect;		/* end sector */
	uint8_t		part_ecyl;		/* end cylinder */
	uint16_t	part_start_lo;		/* absolute starting ... */
	uint16_t	part_start_hi;		/* ... sector number */
	uint16_t	part_size_lo;		/* partition size ... */
	uint16_t	part_size_hi;		/* ... in sectors */
};

struct mbr {
	uint16_t	mbr_code[223];
	struct mbr_part	mbr_part[4];
	uint16_t	mbr_sig;
#define	MBR_SIG		0xAA55
};

extern char *device_name;
extern off_t mediasz;
extern u_int parts;
extern u_int secsz;
extern int readonly, verbose;

uint32_t crc32(const void *, size_t);
map_t	*gpt_add_part(int, uuid_t *, off_t, off_t, unsigned int *);
void	gpt_close(int);
int	gpt_open(const char *);
void*	gpt_read(int, off_t, size_t);
int	gpt_write(int, map_t *);

uint8_t *utf16_to_utf8(uint16_t *);
void	utf8_to_utf16(const uint8_t *, uint16_t *, size_t);

int	cmd_add(int, char *[]);
int	cmd_boot(int, char *[]);
int	cmd_create(int, char *[]);
int	cmd_destroy(int, char *[]);
int	cmd_label(int, char *[]);
int	cmd_migrate(int, char *[]);
int	cmd_recover(int, char *[]);
int	cmd_remove(int, char *[]);
int	cmd_show(int, char *[]);

#endif /* _GPT_H_ */
