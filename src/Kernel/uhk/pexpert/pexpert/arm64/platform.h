/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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
#ifndef _PEXPERT_ARM64_PLATFORM_H_
#define _PEXPERT_ARM64_PLATFORM_H_

/*
 * EmbeddedHeaders defines required: SPDS_ENABLE_STRUCTS and SPDS_ENABLE_ENUMS.
 */

#ifndef SPDS_ENABLE_STRUCTS
#define SPDS_ENABLE_STRUCTS                     1       // Enable structure definitions
#endif /* SPDS_ENABLE_STRUCTS */
#ifndef SPDS_ENABLE_ENUMS
#define SPDS_ENABLE_ENUMS                       1       // Enable enumeration definitions
#endif /* SPDS_ENABLE_ENUMS */

#pragma mark EmbeddedHeaders include macros
/*
 * Define a macro to construct an include path for a sub-platform file.
 * Example: #include SUB_PLATFORM_SPDS_HEADER(p_acc)
 * where ARM64_SOC_NAME is txxxx, and where SPDS_CHIP_REV_LC is a0
 * Expands: #include <soc/txxxx/a0/module/p_acc.h>
 * Lifted and adapted from iBoot/platform.h.
 */
#define NOQUOTE(x) x
#define COMBINE3(a, b, c)                       NOQUOTE(a)NOQUOTE(b)NOQUOTE(c)
#define COMBINE5(a, b, c, d, e)                 NOQUOTE(a)NOQUOTE(b)NOQUOTE(c)NOQUOTE(d)NOQUOTE(e)
#define COMBINE7(a, b, c, d, e, f, g)           NOQUOTE(a)NOQUOTE(b)NOQUOTE(c)NOQUOTE(d)NOQUOTE(e)NOQUOTE(f)NOQUOTE(g)

#define SUB_PLATFORM_HEADER(x)                  <COMBINE5(platform/,x,_,ARM64_SOC_NAME,.h)>
#define SUB_PLATFORM_SOC_HEADER(x)              <COMBINE5(platform/soc/,x,_,ARM64_SOC_NAME,.h)>
#define SUB_PLATFORM_NONMODULE_HEADER(x)        <COMBINE5(soc/,PLATFORM_SPDS_CHIP_REV_LC,/,x,.h)>
#define SUB_PLATFORM_SPDS_HEADER(x)             <COMBINE5(soc/,PLATFORM_SPDS_CHIP_REV_LC,/module/,x,.h)>
#define SUB_PLATFORM_TARGET_HEADER(x)           <COMBINE5(target/,x,_,ARM64_SOC_NAME,.h)>
#define SUB_PLATFORM_TUNABLE_HEADER(r, x)       <COMBINE7(platform/soc/tunables/,ARM64_SOC_NAME,/,r,/,x,.h)>
#define SUB_TARGET_TUNABLE_HEADER(r, t, x)      <COMBINE7(target/tunables/,t,/,r,/,x,.h)>

#ifndef ARM64_SOC_NAME
#ifndef CURRENT_MACHINE_CONFIG_LC
#error CURRENT_MACHINE_CONFIG_LC must be defined in makedefs/MakeInc.def
#endif /* CURRENT_MACHINE_CONFIG_LC */
#define ARM64_SOC_NAME CURRENT_MACHINE_CONFIG_LC
#endif /* ARM64_SOC_NAME */

#define SPDS_CHIP_REV_LC latest
#define PLATFORM_SPDS_CHIP_REV_LC ARM64_SOC_NAME/SPDS_CHIP_REV_LC

#endif /* !_PEXPERT_ARM64_PLATFORM_H_ */
