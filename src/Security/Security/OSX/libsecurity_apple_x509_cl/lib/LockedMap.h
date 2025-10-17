/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
 * LockedMap.h - STL-style map with attached Mutex
 *
 * Copyright (c) 2000,2011,2014 Apple Inc. 
 */
 
#ifndef	_LOCKED_MAP_H_
#define _LOCKED_MAP_H_

#include <map>
#include <security_utilities/threading.h>

template <class KeyType, class ValueType>
class LockedMap
{
private:
	typedef std::map<KeyType, ValueType *> MapType;
	MapType					mMap;
	Mutex					mMapLock;
	
	/* low-level lookup, cacheMapLock held on entry and exit */
	ValueType 				
	*lookupEntryLocked(KeyType key) 
		{
			// don't create new entry if desired entry isn't there
			typename MapType::iterator it = mMap.find(key);
			if(it == mMap.end()) {
				return NULL;
			}
			return it->second;
		}

public:
	/* high level maintenance */
	void 
	addEntry(ValueType &value, KeyType key)
		{
			StLock<Mutex> _(mMapLock);
			mMap[key] = &value;
		}
		
	ValueType				
	*lookupEntry(KeyType key)
		{
			StLock<Mutex> _(mMapLock);
			return lookupEntryLocked(key);
		}
		
	void	
	removeEntry(KeyType key)
		{
			StLock<Mutex> _(mMapLock);

			ValueType *value = lookupEntryLocked(key);
			if(value != NULL) {
				mMap.erase(key);
			}
		}
		
	ValueType	
	*removeFirstEntry()
		{
			StLock<Mutex> _(mMapLock);
			typename MapType::iterator it = mMap.begin();
			if(it == mMap.end()) {
				return NULL;
			}
			ValueType *rtn = it->second;
			mMap.erase(it->first);
			return rtn;
		}
};

#endif	/* _LOCKED_MAP_H_ */
