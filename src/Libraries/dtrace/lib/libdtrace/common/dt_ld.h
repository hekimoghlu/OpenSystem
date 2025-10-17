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
/*
 * Copyright 2006 Apple Computer, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef	_DT_LD_H
#define	_DT_LD_H

#include <libctf.h>
#include <dtrace.h>

 #ifdef __cplusplus
 extern "C" {
 #endif
	
void* dtrace_ld_create_dof(cpu_type_t cpu,             // [provided by linker] target architecture
                           unsigned int typeCount,     // [provided by linker] number of stability or typedef symbol names
                           const char* typeNames[],    // [provided by linker] stability or typedef symbol names
                           unsigned int probeCount,    // [provided by linker] number of probe or isenabled locations
                           const char* probeNames[],   // [provided by linker] probe or isenabled symbol names
                           const char* probeWithin[],  // [provided by linker] function name containing probe or isenabled
                           uint64_t offsetsInDOF[],    // [allocated by linker, populated by DTrace] per-probe offset in the DOF
                           size_t* size);               // [allocated by linker, populated by DTrace] size of the DOF)

char* dt_ld_encode_stability(char* provider_name, dt_provider_t* provider);
char* dt_ld_encode_typedefs(char* provider_name, dt_provider_t* provider);
char* dt_ld_encode_probe(char* provider_name, char* probe_name, dt_probe_t* probe);
char* dt_ld_encode_isenabled(char* provider_name, char* probe_name);

#ifdef __cplusplus
}
#endif

#endif	/* _DT_LD_H */
