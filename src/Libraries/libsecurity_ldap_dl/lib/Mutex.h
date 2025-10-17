/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#ifndef __MUTEX_H__
#define __MUTEX_H__



#include <pthread.h>



// base class for a mutex -- note that this can't be instantiated
class Mutex
{
protected:
	pthread_mutex_t *mMutexPtr;
	Mutex () {}

public:
	void Lock ();
	void Unlock ();
};




// Mutex which initializes its own mutex
class DynamicMutex : public Mutex
{
protected:
	pthread_mutex_t mMutex;

public:
	DynamicMutex ();
	~DynamicMutex ();
};



// Mutex which takes an externally initialized mutex
class StaticMutex : public Mutex
{
protected:
	pthread_mutex_t& mMutex;

public:
	StaticMutex (pthread_mutex_t &mutex);
};



// class which locks and unlocks a mutex when it goes in and out of scope
class MutexLocker
{
protected:
	Mutex& mMutex;

public:
	MutexLocker (Mutex &mutex);
	~MutexLocker ();
};



#endif
