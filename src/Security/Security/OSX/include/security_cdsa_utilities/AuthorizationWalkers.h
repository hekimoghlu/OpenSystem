/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
 *  AuthorizationWalkers.h
 *  SecurityCore
 */

#if !defined(__AuthorizationWalkers__)
#define __AuthorizationWalkers__ 1

#include <Security/Authorization.h>
#include <Security/AuthorizationPlugin.h>
#include <security_cdsa_utilities/walkers.h>
#include <security_cdsa_utilities/cssmwalkers.h> // char * walker

namespace Security {
namespace DataWalkers {


template <class Action>
void walk(Action &operate, AuthorizationItem &item)
{
	operate(item);
	walk(operate, const_cast<char *&>(item.name));
	operate.blob(item.value, item.valueLength);
	// Ignore reserved
}

template <class Action>
AuthorizationItemSet *walk(Action &operate, AuthorizationItemSet * &itemSet)
{
	operate(itemSet);
	operate.blob(itemSet->items, itemSet->count * sizeof(itemSet->items[0]));
	for (uint32 n = 0; n < itemSet->count; n++)
		walk(operate, itemSet->items[n]);
	return itemSet;
}

template <class Action>
void walk(Action &operate, AuthorizationValue &authvalue)
{
    operate.blob(authvalue.data, authvalue.length);
}

template <class Action>
AuthorizationValueVector *walk(Action &operate, AuthorizationValueVector * &valueVector)
{
    operate(valueVector);
    operate.blob(valueVector->values, valueVector->count * sizeof(valueVector->values[0]));
    for (uint32 n = 0; n < valueVector->count; n++)
        walk(operate, valueVector->values[n]);
    return valueVector;
}



} // end namespace DataWalkers
} // end namespace Security

#endif /* ! __AuthorizationWalkers__ */
