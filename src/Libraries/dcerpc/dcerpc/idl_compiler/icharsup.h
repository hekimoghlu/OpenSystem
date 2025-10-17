/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
**
**  NAME:
**
**      icharsup.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Function prototypes and type definitions for international
**          character support
**
**
*/

#ifndef ICHARSUP_H
#define ICHARSUP_H

/* Description of an operation's use of I-char machinery */
typedef struct BE_cs_info_t {
    boolean cs_machinery;   /* TRUE if operation has I-char machinery */
    boolean stag_by_ref;    /* TRUE if passed by ref according to IDL */
    NAMETABLE_id_t  stag;
    boolean drtag_by_ref;    /* TRUE if passed by ref according to IDL */
    NAMETABLE_id_t  drtag;
    boolean rtag_by_ref;    /* TRUE if passed by ref according to IDL */
    NAMETABLE_id_t  rtag;
} BE_cs_info_t;

void BE_cs_analyze_and_spell_vars
(
    FILE *fid,                      /* [in] Handle for emitted C text */
    AST_operation_n_t *p_operation, /* [in] Pointer to AST operation node */
    BE_side_t side,                 /* [in] client or server */
    BE_cs_info_t *p_cs_info         /* [out] Description of I-char machinery */
);

void BE_spell_cs_state
(
    FILE *fid,                      /* [in] Handle for emitted C text */
    const char *state_access,       /* [in] "IDL_ms." or "IDL_msp->" */
    BE_side_t side,                 /* [in] client or server */
    BE_cs_info_t *p_cs_info         /* [in] Description of I-char machinery */
);

void BE_spell_cs_tag_rtn_call
(
    FILE *fid,                      /* [in] Handle for emitted C text */
    const char *state_access,       /* [in] "IDL_ms." or "IDL_msp->" */
    AST_operation_n_t *p_operation, /* [in] Pointer to AST operation node */
    BE_side_t side,                 /* [in] client or server */
    BE_handle_info_t *p_handle_info,/* [in] How to spell binding handle name */
    BE_cs_info_t *p_cs_info,        /* [in] Description of I-char machinery */
    boolean pickling
);

#endif
