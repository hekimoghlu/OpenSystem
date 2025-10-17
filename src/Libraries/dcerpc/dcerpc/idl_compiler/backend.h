/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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
**      backend.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**  Header for backend.c
**
**  VERSION: DCE 1.0
**
*/
boolean BE_main
(
    boolean             *cmd_opt,   /* [in] array of cmd option flags */
    void                **cmd_val,  /* [in] array of cmd option values */
    FILE                *h_fid,     /* [in] header file handle, or NULL */
    FILE                *caux_fid,  /* [in] client aux file handle, or NULL */
    FILE                *saux_fid,  /* [in] server aux file handle, or NULL */
    FILE                *cstub_fid, /* [in] cstub file handle, or NULL */
    FILE                *sstub_fid, /* [in] sstub file handle, or NULL */
    AST_interface_n_t   *int_p      /* [in] ptr to interface node */
);

void BE_push_malloc_ctx
(
      void
);

void BE_push_perm_malloc_ctx
(
      void
);

void BE_pop_malloc_ctx
(
      void
);

heap_mem *BE_ctx_malloc
(
      size_t size
);
