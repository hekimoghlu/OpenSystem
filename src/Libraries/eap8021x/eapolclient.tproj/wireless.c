/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
 * wireless.c
 */

/* 
 * Modification History
 *
 * October 26, 2001	Dieter Siegmund (dieter@apple)
 * - created
 */

#include <TargetConditionals.h>

#ifndef TEST_WIRELESS
#if ! TARGET_OS_IPHONE
#include "my_darwin.h"
#endif /* TARGET_OS_IPHONE */
#endif /* TEST_WIRELESS */

#include <SystemConfiguration/SCValidation.h>
#include <SystemConfiguration/SCPrivate.h>
#include "EAPLog.h"

