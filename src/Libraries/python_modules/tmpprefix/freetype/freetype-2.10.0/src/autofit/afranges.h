/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#ifndef AFRANGES_H_
#define AFRANGES_H_


#include "aftypes.h"


FT_BEGIN_HEADER

#undef  SCRIPT
#define SCRIPT( s, S, d, h, H, ss )                                     \
          extern const AF_Script_UniRangeRec  af_ ## s ## _uniranges[];

#include "afscript.h"

#undef  SCRIPT
#define SCRIPT( s, S, d, h, H, ss )                                             \
          extern const AF_Script_UniRangeRec  af_ ## s ## _nonbase_uniranges[];

#include "afscript.h"

 /* */

FT_END_HEADER

#endif /* AFRANGES_H_ */


/* END */
