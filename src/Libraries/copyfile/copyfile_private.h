/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#ifndef _COPYFILE_PRIVATE_H
# define _COPYFILE_PRIVATE_H

/*
 * Set (or get) the intent type; see xattr_properties.h for details.
 * This command uses a pointer to CopyOperationIntent_t as the parameter.
 */
# define COPYFILE_STATE_INTENT	256

/*
 * File flags that are not preserved when copying stat information.
 */
#define COPYFILE_OMIT_FLAGS 	(UF_TRACKED | SF_RESTRICTED | SF_NOUNLINK | UF_DATAVAULT)

/*
 * File flags that are not removed when replacing an existing file.
 */
#define COPYFILE_PRESERVE_FLAGS	(SF_RESTRICTED | SF_NOUNLINK | UF_DATAVAULT)

#endif /* _COPYFILE_PRIVATE_H */
