/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#include "Mutex.h"


StaticMutex::StaticMutex (pthread_mutex_t &ref) : mMutex (ref)
{
	mMutexPtr = &ref;
}



DynamicMutex::DynamicMutex ()
{
	pthread_mutex_init (&mMutex, NULL);
	mMutexPtr = &mMutex;
}



DynamicMutex::~DynamicMutex ()
{
	pthread_mutex_destroy (&mMutex);
}



void Mutex::Lock ()
{
	pthread_mutex_lock (mMutexPtr);
}



void Mutex::Unlock ()
{
	pthread_mutex_unlock (mMutexPtr);
}




MutexLocker::MutexLocker (Mutex &mutex) : mMutex (mutex)
{
	mMutex.Lock ();
}



MutexLocker::~MutexLocker ()
{
	mMutex.Unlock ();
}
