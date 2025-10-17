/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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
#ifndef _EAPOLCFG_AUTH_TYPES_H
#define _EAPOLCFG_AUTH_TYPES_H

/*
 * Keep IPC functions private to the framework
 */
#ifdef mig_external
#undef mig_external
#endif
#define mig_external __private_extern__

#if 0
/* Turn MIG type checking on by default */
#ifdef __MigTypeCheck
#undef __MigTypeCheck
#endif
#define __MigTypeCheck	1
#endif /* 0 */

/*
 * Mach server port name
 */
#define EAPOLCFG_AUTH_SERVER	"com.apple.eapolcfg_auth"

enum {
    keapolcfg_auth_set_name		= 0x1,
    keapolcfg_auth_set_password		= 0x2
};
typedef const char * xmlData_t;
typedef const char * OOBData_t;
typedef char * OOBDataOut_t;

#endif /* _EAPOLCFG_AUTH_TYPES_H */
