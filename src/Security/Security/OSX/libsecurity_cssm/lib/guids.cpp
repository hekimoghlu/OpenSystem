/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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
#include <Security/cssmapple.h>
#include <libkern/OSByteOrder.h>

// {87191ca0-0fc9-11d4-849a-000502b52122}
const CSSM_GUID gGuidCssm =
{
	OSSwapHostToBigConstInt32(0x87191ca0),
	OSSwapHostToBigConstInt16(0x0fc9),
	OSSwapHostToBigConstInt16(0x11d4),
	{ 0x84, 0x9a, 0x00, 0x05, 0x02, 0xb5, 0x21, 0x22 }
};

// {87191ca1-0fc9-11d4-849a-000502b52122}
const CSSM_GUID gGuidAppleFileDL =
{
	OSSwapHostToBigConstInt32(0x87191ca1),
	OSSwapHostToBigConstInt16(0x0fc9),
	OSSwapHostToBigConstInt16(0x11d4),
	{ 0x84, 0x9a, 0x00, 0x05, 0x02, 0xb5, 0x21, 0x22 }
};

// {87191ca2-0fc9-11d4-849a-000502b52122}
const CSSM_GUID gGuidAppleCSP =
{
	OSSwapHostToBigConstInt32(0x87191ca2),
	OSSwapHostToBigConstInt16(0x0fc9),
	OSSwapHostToBigConstInt16(0x11d4),
	{ 0x84, 0x9a, 0x00, 0x05, 0x02, 0xb5, 0x21, 0x22 }
};

// {87191ca3-0fc9-11d4-849a-000502b52122}
const CSSM_GUID gGuidAppleCSPDL =
{
	OSSwapHostToBigConstInt32(0x87191ca3), 
	OSSwapHostToBigConstInt16(0x0fc9),
	OSSwapHostToBigConstInt16(0x11d4),
	{ 0x84, 0x9a, 0x00, 0x05, 0x02, 0xb5, 0x21, 0x22 }
};

// {87191ca4-0fc9-11d4-849a-000502b52122}
const CSSM_GUID gGuidAppleX509CL =
{
	OSSwapHostToBigConstInt32(0x87191ca4),
	OSSwapHostToBigConstInt16(0x0fc9),
	OSSwapHostToBigConstInt16(0x11d4),
	{ 0x84, 0x9a, 0x00, 0x05, 0x02, 0xb5, 0x21, 0x22 }
};

// {87191ca5-0fc9-11d4-849a-000502b52122}
const CSSM_GUID gGuidAppleX509TP =
{
	OSSwapHostToBigConstInt32(0x87191ca5),
	OSSwapHostToBigConstInt16(0x0fc9),
	OSSwapHostToBigConstInt16(0x11d4),
	{ 0x84, 0x9a, 0x00, 0x05, 0x02, 0xb5, 0x21, 0x22 }
};

// {87191ca6-0fc9-11d4-849a-000502b52122}
const CSSM_GUID gGuidAppleLDAPDL =
{
	OSSwapHostToBigConstInt32(0x87191ca6),
	OSSwapHostToBigConstInt16(0x0fc9),
	OSSwapHostToBigConstInt16(0x11d4),
	{ 0x84, 0x9a, 0x00, 0x05, 0x02, 0xb5, 0x21, 0x22 }
};

// {87191ca7-0fc9-11d4-849a-000502b52122}
const CSSM_GUID gGuidAppleDotMacTP =
{
	OSSwapHostToBigConstInt32(0x87191ca7),
	OSSwapHostToBigConstInt16(0x0fc9),
	OSSwapHostToBigConstInt16(0x11d4),
	{ 0x84, 0x9a, 0x00, 0x05, 0x02, 0xb5, 0x21, 0x22 }
};

// 87191ca8-0fc9-11d4-849a000502b52122
const CSSM_GUID gGuidAppleSdCSPDL =
{
	OSSwapHostToBigConstInt32(0x87191ca8),
	OSSwapHostToBigConstInt16(0x0fc9),
	OSSwapHostToBigConstInt16(0x11d4),
	{ 0x84, 0x9a, 0x00, 0x05, 0x02, 0xb5, 0x21, 0x22 }
};

// {87191ca9-0fc9-11d4-849a-000502b52122}
const CSSM_GUID gGuidAppleDotMacDL =
{
	OSSwapHostToBigConstInt32(0x87191ca9),
	OSSwapHostToBigConstInt16(0x0fc9),
	OSSwapHostToBigConstInt16(0x11d4),
	{ 0x84, 0x9a, 0x00, 0x05, 0x02, 0xb5, 0x21, 0x22 }
};
