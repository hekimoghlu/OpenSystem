/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#ifndef CommonNumerics_crc_h
#define CommonNumerics_crc_h


#if defined (_WIN32)
#define __unused
#endif

#include <stdint.h>
#include <stddef.h>
#include "../lib/ccDispatch.h"

#define MASK08 0x00000000000000ffLL
#define MASK16 0x000000000000ffffLL
#define MASK32 0x00000000ffffffffLL
#define MASK64 0xffffffffffffffffLL



#define WEAK_CHECK_INPUT "123456789"

// Utility Functions

uint8_t reflect_byte(uint8_t b);
uint64_t reflect(uint64_t w, size_t bits);
uint64_t reverse_poly(uint64_t poly, size_t width);

typedef uint64_t (*cccrc_setup_p)(void);
typedef uint64_t (*cccrc_update_p)(size_t len, const void *in, uint64_t crc);
typedef uint64_t (*cccrc_final_p)(size_t length, uint64_t crc);
typedef uint64_t (*cccrc_oneshot_p)(size_t len, const void *in);

#define NO_REFLECT_REVERSE 0
#define REFLECT_IN 1
#define REVERSE_OUT 2
#define REFLECT_REVERSE 3

typedef struct crcModelParms_t {
    int width; // width in bytes
    int reflect_reverse;
    uint64_t mask;
    uint64_t poly;
    uint64_t initial_value;
    uint64_t final_xor;
    uint64_t weak_check;
} crcModelParms;

typedef struct crcFuncs_t {
    cccrc_setup_p setup;
    cccrc_update_p update;
    cccrc_final_p final;
    cccrc_oneshot_p oneshot;
} crcFuncs;

enum crcType_t {
    model = 0,
    functions = 1,
};
typedef uint32_t crcType;

typedef struct crcDescriptor_t {
    const char *name;
    const crcType defType;
    union ddef {
        crcModelParms parms;
        crcFuncs funcs;
    } def;
} crcDescriptor;

typedef const crcDescriptor *crcDescriptorPtr;

typedef struct crcInfo_t {
    dispatch_once_t table_init;
    crcDescriptorPtr descriptor;
    size_t size;
    union {
        uint8_t *bytes;
        uint16_t *b16;
        uint32_t *b32;
        uint64_t *b64;
    } table;
} crcInfo, *crcInfoPtr;


void gen_std_crc_table(void *c);
void dump_crc_table(crcInfoPtr crc);
uint64_t crc_normal_init(crcInfoPtr crc);
uint64_t crc_normal_update(crcInfoPtr crc, uint8_t *p, size_t len, uint64_t current);
uint64_t crc_normal_final(crcInfoPtr crc, uint64_t current);
uint64_t crc_normal_oneshot(crcInfoPtr crc, uint8_t *p, size_t len);

uint64_t crc_reverse_init(crcInfoPtr crc);
uint64_t crc_reverse_update(crcInfoPtr crc, uint8_t *p, size_t len, uint64_t current);
uint64_t crc_reverse_final(crcInfoPtr crc, uint64_t current);
uint64_t crc_reverse_oneshot(crcInfoPtr crc, uint8_t *p, size_t len);

static inline uint64_t descmaskfunc(crcDescriptorPtr descriptor) {
    switch(descriptor->def.parms.width) {
        case 1: return MASK08;
        case 2: return MASK16;
        case 4: return MASK32;
        case 8: return MASK64;
    }
    return 0;
}

extern const crcDescriptor CC_crc8;
extern const crcDescriptor CC_crc8_icode;
extern const crcDescriptor CC_crc8_itu;
extern const crcDescriptor CC_crc8_rohc;
extern const crcDescriptor CC_crc8_wcdma;
extern const crcDescriptor CC_crc16;
extern const crcDescriptor CC_crc16_ccitt_true;
extern const crcDescriptor CC_crc16_ccitt_false;
extern const crcDescriptor CC_crc16_usb;
extern const crcDescriptor CC_crc16_xmodem;
extern const crcDescriptor CC_crc16_dect_r;
extern const crcDescriptor CC_crc16_dect_x;
extern const crcDescriptor CC_crc16_icode;
extern const crcDescriptor CC_crc16_verifone;
extern const crcDescriptor CC_crc16_a;
extern const crcDescriptor CC_crc16_b;
extern const crcDescriptor CC_crc32;
extern const crcDescriptor CC_crc32_castagnoli;
extern const crcDescriptor CC_crc32_bzip2;
extern const crcDescriptor CC_crc32_mpeg_2;
extern const crcDescriptor CC_crc32_posix;
extern const crcDescriptor CC_crc32_xfer;
extern const crcDescriptor CC_adler32;
extern const crcDescriptor CC_crc64_ecma_182;

#endif
