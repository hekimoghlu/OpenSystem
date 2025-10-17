/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#include <stdio.h>
#include <CommonNumerics/CommonNumerics.h>
#include <CommonNumerics/CommonCRC.h>
#include "crc.h"
#include "../lib/ccGlobals.h"
#include <stdlib.h>

static inline crcInfoPtr getDesc(CNcrc algorithm)
{
	cc_globals_t globals = _cc_globals();
    return &globals->crcSelectionTab[algorithm];
}

typedef struct crcRef_int {
    crcInfoPtr crc;
    uint64_t current;
    size_t length;
} *crcRefptr;

static inline uint64_t
try_generic_oneshot(crcInfoPtr crc, size_t len, const void *in)
{
    if(crc->descriptor->def.parms.reflect_reverse) return crc_reverse_oneshot(crc, (uint8_t *) in, len);
    else return crc_normal_oneshot(crc, (uint8_t *) in, len);
}

static inline uint64_t
try_generic_setup(crcInfoPtr crc)
{
    if(crc->descriptor->def.parms.reflect_reverse) return crc_reverse_init(crc);
    else return crc_normal_init(crc);
}

static inline uint64_t
try_generic_update(crcInfoPtr crc, size_t len, const void *in, uint64_t current)
{
    if(crc->descriptor->def.parms.reflect_reverse)
        return crc_reverse_update(crc, (uint8_t *) in, len, current);
    else
        return crc_normal_update(crc, (uint8_t *) in, len, current);
}

static inline uint64_t
try_generic_final(crcInfoPtr crc, uint64_t current)
{
    if(crc->descriptor->def.parms.reflect_reverse) return crc_reverse_final(crc, current);
    else return crc_normal_final(crc, current);
}

CNStatus
CNCRC(CNcrc algorithm, const void *in, size_t len, uint64_t *result)
{
    crcInfoPtr crc = getDesc(algorithm);
    if(crc->descriptor == NULL) return kCNUnimplemented;
    if(crc->descriptor->defType == model)
        *result = try_generic_oneshot(crc, len, in);
    else
        *result = crc->descriptor->def.funcs.oneshot(len, in);
    return kCNSuccess;
}

CNStatus
CNCRCInit(CNcrc algorithm, CNCRCRef *crcRef)
{
    crcRefptr retval = malloc(sizeof(struct crcRef_int));
    if(retval == NULL) return kCNMemoryFailure;
    retval->crc = getDesc(algorithm);
    if(retval->crc->descriptor == NULL) {
        free(retval);
        return kCNUnimplemented;
    }
    retval->current = 0;
    retval->length = 0;
    if(retval->crc->descriptor->defType == model) retval->current = try_generic_setup(retval->crc);
    else retval->current = retval->crc->descriptor->def.funcs.setup();
    *crcRef = (CNCRCRef) retval;
    return kCNSuccess;
}


CNStatus
CNCRCRelease(CNCRCRef crcRef)
{
    crcRefptr ref = (crcRefptr) crcRef;
    free(ref);
    return kCNSuccess;
}

CNStatus
CNCRCUpdate(CNCRCRef crcRef, const void *in, size_t len)
{
    crcRefptr ref = (crcRefptr) crcRef;
    if(ref->crc->descriptor->defType == functions) ref->current = ref->crc->descriptor->def.funcs.update(len, in, ref->current);
    else ref->current = try_generic_update(ref->crc, len, in, ref->current);
    ref->length += len;
    return kCNSuccess;
}


CNStatus
CNCRCFinal(CNCRCRef crcRef, uint64_t *result)
{
    crcRefptr ref = (crcRefptr) crcRef;
    if(ref->crc->descriptor->defType == functions) ref->current = ref->crc->descriptor->def.funcs.final(ref->length, ref->current);
    else ref->current = try_generic_final(ref->crc, ref->current);
    *result = ref->current;
    return kCNSuccess;
}

CNStatus
CNCRCDumpTable(CNcrc algorithm)
{
    crcInfoPtr crc = getDesc(algorithm);
    if(crc->descriptor == NULL) return kCNUnimplemented;
    if(crc->descriptor->defType != model) return kCNParamError;
    (void) try_generic_setup(crc);
    dump_crc_table(crc);
    return kCNSuccess;
    
}

CNStatus
CNCRCWeakTest(CNcrc algorithm)
{
    crcInfoPtr crc = getDesc(algorithm);    
    if(!crc->descriptor || crc->descriptor->defType == functions) return kCNSuccess;
    
    uint64_t result = try_generic_oneshot(crc, 9, "123456789");
    if(result == crc->descriptor->def.parms.weak_check) return kCNSuccess;
    return kCNFailure;
}

