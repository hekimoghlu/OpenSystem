/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#ifndef _DBQUERY_H_
#define _DBQUERY_H_  1

#include <security_cdsa_utilities/handleobject.h>

#ifdef _CPP_DBQUERY
# pragma export on
#endif

namespace Security
{

class DbQuery: public HandleObject
{
    NOCOPY(DbQuery);
public:
    DbQuery ();
    virtual ~DbQuery ();
};

} // end namespace Security

#ifdef _CPP_DBQUERY
# pragma export off
#endif

#endif // _DBQUERY_H_
