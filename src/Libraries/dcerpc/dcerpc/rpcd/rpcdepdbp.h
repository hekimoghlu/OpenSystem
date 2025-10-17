/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
**      rpcdepdbp.h
**
**  FACILITY:
**
**      RPC Daemon
**
**  ABSTRACT:
**
**      RPCD Endpoint Database Mgmt - routines, etc. shared by modules
**      which know more about epdb internals
**
**
**
*/

#ifndef RPCDEPDBP_H
#define RPCDEPDBP_H

/*
 *  Delete disk copy of entry and free associated
 *  memory
 */
PRIVATE void epdb_delete_entry
    (
        struct db       *h,
        db_entry_p_t    entp,
        error_status_t  *status
    );

PRIVATE void sliv_init
    (
        struct db       *h,
        error_status_t  *status
    );

#endif
