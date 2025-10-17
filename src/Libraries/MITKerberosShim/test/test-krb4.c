/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#include "test.h"

OSStatus KClientGetVersion (UInt16*, UInt16*, const char**);
OSErr KPPreferencesFileIsWritable(const FSSpec *);
int put_svc_key(char*,char*,char*,char*,int, char*);

int main(int argc, char **argv)
{

	VERIFY_DEPRECATED_I(
		"KClientGetVersion",
		KClientGetVersion(0,0,NULL));

	VERIFY_DEPRECATED_I(
		"KPPreferencesFileIsWritable",
		KPPreferencesFileIsWritable(NULL));

	VERIFY_DEPRECATED_I(
		"put_svc_key",
		put_svc_key(NULL, NULL, NULL, NULL, 0, NULL));

	return 0;
}
