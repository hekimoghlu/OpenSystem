/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
/* Defines the macro RTNAME(n) which decorates the external name of a runtime
 * library function or object with extra characters so that it
 * (a) is not in the user's name space,
 * (b) doesn't conflict with other libraries, and
 * (c) prevents incompatible versions of the runtime library from linking
 *
 * The value of REVISION should not be changed until/unless the API to the
 * runtime library must change in some way that breaks backward compatibility.
 */
#ifndef LANGUAGE_COMPABILITY_RUNTIME_ENTRY_NAMES_H
#define LANGUAGE_COMPABILITY_RUNTIME_ENTRY_NAMES_H

#include "language/Compability/Common/api-attrs.h"

#ifndef RTNAME
#define NAME_WITH_PREFIX_AND_REVISION(prefix, revision, name) \
  prefix##revision##name
#define RTNAME(name) NAME_WITH_PREFIX_AND_REVISION(_Fortran, A, name)
#endif

#ifndef RTDECL
#define RTDECL(name) RT_API_ATTRS RTNAME(name)
#endif

#ifndef RTDEF
#define RTDEF(name) RT_API_ATTRS RTNAME(name)
#endif

#ifndef RTNAME_STRING
#define RTNAME_STRINGIFY_(x) #x
#define RTNAME_STRINGIFY(x) RTNAME_STRINGIFY_(x)
#define RTNAME_STRING(name) RTNAME_STRINGIFY(RTNAME(name))
#endif

#endif /* !FORTRAN_RUNTIME_ENTRY_NAMES_H */
