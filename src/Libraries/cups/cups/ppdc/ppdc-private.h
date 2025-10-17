/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
// Private definitions for the CUPS PPD Compiler.
//
// Copyright 2009-2010 by Apple Inc.
//
// Licensed under Apache License v2.0.  See the file "LICENSE" for more information.
//

#ifndef _PPDC_PRIVATE_H_
#  define _PPDC_PRIVATE_H_

//
// Include necessary headers...
//

#  include "ppdc.h"
#  include <cups/cups-private.h>


//
// Macros...
//

#  ifdef PPDC_DEBUG
#    define PPDC_NEW		DEBUG_printf(("%s: %p new", class_name(), this))
#    define PPDC_NEWVAL(s)	DEBUG_printf(("%s(\"%s\"): %p new", class_name(), s, this))
#    define PPDC_DELETE		DEBUG_printf(("%s: %p delete", class_name(), this))
#    define PPDC_DELETEVAL(s)	DEBUG_printf(("%s(\"%s\"): %p delete", class_name(), s, this))
#  else
#    define PPDC_NEW
#    define PPDC_NEWVAL(s)
#    define PPDC_DELETE
#    define PPDC_DELETEVAL(s)
#  endif /* PPDC_DEBUG */

#endif // !_PPDC_PRIVATE_H_
