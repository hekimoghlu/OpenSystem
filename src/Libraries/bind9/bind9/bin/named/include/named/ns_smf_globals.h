/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
/* $Id: ns_smf_globals.h,v 1.7 2007/06/19 23:46:59 tbox Exp $ */

#ifndef NS_SMF_GLOBALS_H
#define NS_SMF_GLOBALS_H 1
 
#include <libscf.h>
 
#undef EXTERN
#undef INIT
#ifdef NS_MAIN
#define EXTERN
#define INIT(v) = (v)
#else
#define EXTERN extern
#define INIT(v)
#endif
                
EXTERN unsigned int	ns_smf_got_instance	INIT(0);
EXTERN unsigned int	ns_smf_chroot		INIT(0);
EXTERN unsigned int	ns_smf_want_disable	INIT(0);

isc_result_t ns_smf_add_message(isc_buffer_t *text);
isc_result_t ns_smf_get_instance(char **name, int debug, isc_mem_t *mctx);

#undef EXTERN
#undef INIT
 
#endif /* NS_SMF_GLOBALS_H */
