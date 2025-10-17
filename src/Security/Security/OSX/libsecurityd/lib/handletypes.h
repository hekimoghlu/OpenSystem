/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
 * handletypes.h 
 */
#ifndef _H_SS_HANDLE_TYPES
#define _H_SS_HANDLE_TYPES

#include <Security/cssmtype.h>
#include <stdint.h>

#ifdef __cplusplus

namespace Security {
namespace SecurityServer {

#endif /* __cplusplus */


/* XXX/gh  Might have to be guarded thusly to protect ss_types.h */
/* #ifndef _H_SS_TYPES */

/*
 * These are all uint32_ts behind the curtain, but we try to be
 * explicit about which kind they are.
 * By protocol, each of these is in a different address space - i.e.
 * a KeyHandle and a DbHandle with the same value may or may not refer
 * to the same thing - it's up to the handle provider.
 * GenericHandle is for cases where a generic handle is further elaborated
 * with a "kind code" - currently for ACL manipulations only.
 */
typedef uint32_t DbHandle;			/* database handle               */
typedef uint32_t KeyHandle;			/* cryptographic key handle      */
typedef uint32_t RecordHandle;		/* data record identifier handle */
typedef uint32_t SearchHandle;		/* search (query) handle         */
typedef uint32_t GenericHandle;		/* for polymorphic handle uses   */

static const DbHandle noDb = 0;
static const KeyHandle noKey = 0;
static const RecordHandle noRecord = 0;
static const SearchHandle noSearch = 0;

/* #endif */  /* _H_SS_TYPES */

/*
 * Required for MIG-generated code; made sense when the above handle types
 * were all CSSM_HANDLEs
 */
typedef uint32_t IPCHandle;
typedef IPCHandle IPCDbHandle;
typedef IPCHandle IPCKeyHandle;
typedef IPCHandle IPCRecordHandle;
typedef IPCHandle IPCSearchHandle;
typedef IPCHandle IPCGenericHandle;

#ifdef __cplusplus

} // end namespace SecurityServer
} // end namespace Security

#endif /* __cplusplus */


#endif /* _H_SS_HANDLE_TYPES */
