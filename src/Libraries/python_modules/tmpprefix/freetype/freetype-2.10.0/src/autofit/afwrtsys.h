/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#ifndef AFWRTSYS_H_
#define AFWRTSYS_H_

  /* Since preprocessor directives can't create other preprocessor */
  /* directives, we have to include the header files manually.     */

#include "afdummy.h"
#include "aflatin.h"
#include "afcjk.h"
#include "afindic.h"
#ifdef FT_OPTION_AUTOFIT2
#include "aflatin2.h"
#endif

#endif /* AFWRTSYS_H_ */


  /* The following part can be included multiple times. */
  /* Define `WRITING_SYSTEM' as needed.                 */


  /* Add new writing systems here.  The arguments are the writing system */
  /* name in lowercase and uppercase, respectively.                      */

  WRITING_SYSTEM( dummy,  DUMMY  )
  WRITING_SYSTEM( latin,  LATIN  )
  WRITING_SYSTEM( cjk,    CJK    )
  WRITING_SYSTEM( indic,  INDIC  )
#ifdef FT_OPTION_AUTOFIT2
  WRITING_SYSTEM( latin2, LATIN2 )
#endif


/* END */
