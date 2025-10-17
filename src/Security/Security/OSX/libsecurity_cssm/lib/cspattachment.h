/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
//
// cspattachment - cryptographic plugin attachment type
//
#ifndef _H_CSPATTACHMENT
#define _H_CSPATTACHMENT

#include "attachment.h"
#include <Security/cssmcspi.h>

#ifdef _CPP_CSPATTACHMENT
# pragma export on
#endif


typedef StandardAttachment<CSSM_SERVICE_CSP, CSSM_SPI_CSP_FUNCS> CSPAttachment;

#ifdef _CPP_CSPATTACHMENT
# pragma export off
#endif

#endif  //_H_CSPATTACHMENT
