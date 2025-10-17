/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
**      rpcdp.h
**
**  FACILITY:
**
**      RPC Daemon
**
**  ABSTRACT:
**
**  RPC Daemon "private" types, defines, ...
**
**
*/

#ifndef RPCDP_H
#define RPCDP_H

/*
** Useful macros
*/

#define STATUS_OK(s) ((s)==NULL || *(s) == rpc_s_ok)
#define SET_STATUS(s,val) *(s) = val
#define SET_STATUS_OK(s) SET_STATUS(s, error_status_ok)
#define STATUS(s) *(s)

EXTERNAL idl_uuid_t nil_uuid;
EXTERNAL boolean32 dflag;

typedef enum {warning, fatal, fatal_usage} check_mode_t;

PRIVATE boolean32 check_st_bad
    (
        const char      *str,
        const error_status_t  *st
    );

PRIVATE void show_st
    (
        const char      *str,
        const error_status_t  *st
    );

#endif
