/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#include "crc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t
cm_tab(crcDescriptorPtr crcdesc, uint8_t index)
{
    uint64_t retval;
    const crcModelParms *desc = &crcdesc->def.parms;
    uint64_t topbit = 1LL << ((desc->width * 8) - 1);
    uint64_t mask = descmaskfunc(crcdesc);

    retval = (desc->reflect_reverse) ?reflect_byte(index): index;
    retval <<= (desc->width*8-8);
    for (int i=0; i<8; i++) {
        if (retval & topbit) retval = (retval << 1) ^ desc->poly;
        else retval <<= 1;
    }
    retval = (desc->reflect_reverse) ?reflect(retval, desc->width*8): retval;
    return retval & mask;
}

void
gen_std_crc_table(void *c)
{
    crcInfoPtr crc = c;
    
    size_t width = crc->descriptor->def.parms.width;
    if((crc->table.bytes = malloc(width * 256)) == NULL) return;
    for(int i=0; i<256; i++){
        uint8_t c8 = i&0xFF;
        switch (width) {
            case 1: crc->table.bytes[i] = (uint8_t) cm_tab(crc->descriptor, c8); break;
            case 2: crc->table.b16[i] = (uint16_t) cm_tab(crc->descriptor, c8); break;
            case 4: crc->table.b32[i] = (uint32_t) cm_tab(crc->descriptor, c8); break;
            case 8: crc->table.b64[i] = (uint64_t) cm_tab(crc->descriptor, c8); break;
        }
    }
    
}

static char * cc_strndup (char const *s, size_t n)
{
	if (s == NULL) return NULL;
    size_t len = strnlen (s, n);
    char *dup = malloc (len + 1);
    
    if (dup == NULL) return NULL;
    
    memcpy(dup, s, len);
    dup [len] = '\0';
    return dup;
}

void
dump_crc_table(crcInfoPtr crc)
{
    size_t width = crc->descriptor->def.parms.width;
    char *name = cc_strndup(crc->descriptor->name, 64);
    int per_line = 8;
    
    for(size_t i=0; i<strlen(name); i++) if(name[i] == '-') name[i] = '_';
    
    switch (width) {
        case 1: printf("const uint8_t %s_crc_table[] = {\n", name); per_line = 16; break;
        case 2: printf("const uint16_t %s_crc_table[] = {\n", name); per_line = 8; break;
        case 4: printf("const uint32_t %s_crc_table[] = {\n", name); per_line = 8; break;
        case 8: printf("const uint64_t %s_crc_table[] = {\n", name); per_line = 4; break;
    }
    
    for(int i=0; i<256; i++) {
        switch (width) {
            case 1: printf(" 0x%02x,", crc->table.bytes[i]); break;
            case 2: printf(" 0x%04x,", crc->table.b16[i]); break;
            case 4: printf(" 0x%08x,", crc->table.b32[i]); break;
            case 8: printf(" 0x%016llx,", crc->table.b64[i]); break;
        }
        if(((i+1) % per_line) == 0) printf("\n");
    }
    printf("};\n\n");
    free(name);
}
