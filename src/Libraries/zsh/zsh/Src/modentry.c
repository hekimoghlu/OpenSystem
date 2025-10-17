/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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

#include "zsh.mdh"

int setup_ _((Module));
int boot_ _((Module));
int cleanup_ _((Module));
int finish_ _((Module));
int modentry _((int boot, Module m, void *ptr));

/**/
int
modentry(int boot, Module m, void *ptr)
{
    switch (boot) {
    case 0:
	return setup_(m);
	break;

    case 1:
	return boot_(m);
	break;

    case 2:
	return cleanup_(m);
	break;

    case 3:
	return finish_(m);
	break;

    case 4:
	return features_(m, (char ***)ptr);
	break;

    case 5:
	return enables_(m, (int **)ptr);
	break;

    default:
	zerr("bad call to modentry");
	return 1;
	break;
    }
}
