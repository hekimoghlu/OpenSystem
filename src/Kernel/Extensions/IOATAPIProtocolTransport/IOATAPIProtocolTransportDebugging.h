/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#ifndef _IOKIT_IO_ATAPI_PROTOCOL_TRANSPORT_DEBUGGING_H_
#define _IOKIT_IO_ATAPI_PROTOCOL_TRANSPORT_DEBUGGING_H_

#include <stdint.h>

#define ATAPI_SYSCTL    "debug.ATAPITransport"

typedef struct ATAPISysctlArgs
{
	uint32_t	type;
	uint32_t	operation;
	uint32_t	debugFlags;
} ATAPISysctlArgs;

#define kATAPITypeDebug			'ATPI'

enum
{
	kATAPIOperationGetFlags 	= 0,
	kATAPIOperationSetFlags 	= 1
};

#endif //_IOKIT_IO_ATAPI_PROTOCOL_TRANSPORT_DEBUGGING_H_
