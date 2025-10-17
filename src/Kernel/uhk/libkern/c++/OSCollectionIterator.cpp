/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
/* IOArray.h created by rsulack on Thu 11-Sep-1997 */

#define IOKIT_ENABLE_SHARED_PTR

#include <libkern/c++/OSArray.h>
#include <libkern/c++/OSCollection.h>
#include <libkern/c++/OSCollectionIterator.h>
#include <libkern/c++/OSLib.h>
#include <libkern/c++/OSSharedPtr.h>

#define super OSIterator

OSDefineMetaClassAndStructors(OSCollectionIterator, OSIterator)

bool
OSCollectionIterator::initWithCollection(const OSCollection *inColl)
{
	if (!super::init() || !inColl) {
		return false;
	}

	collection.reset(inColl, OSRetain);
	collIterator = NULL;
	initialUpdateStamp = 0;
	valid = false;

	return true;
}

OSSharedPtr<OSCollectionIterator>
OSCollectionIterator::withCollection(const OSCollection *inColl)
{
	OSSharedPtr<OSCollectionIterator> me = OSMakeShared<OSCollectionIterator>();

	if (me && !me->initWithCollection(inColl)) {
		return nullptr;
	}

	return me;
}

void
OSCollectionIterator::free()
{
	freeIteratorStorage();

	collection.reset();

	super::free();
}

void
OSCollectionIterator::reset()
{
	valid = false;
	bool initialized = initializeIteratorStorage();

	if (!initialized) {
		// reusing existing storage
		void * storage = getIteratorStorage();
		bzero(storage, collection->iteratorSize());

		if (!collection->initIterator(storage)) {
			return;
		}

		initialUpdateStamp = collection->updateStamp;
		valid = true;
	}
}

bool
OSCollectionIterator::isValid()
{
	initializeIteratorStorage();

	if (!valid || collection->updateStamp != initialUpdateStamp) {
		return false;
	}

	return true;
}

bool
OSCollectionIterator::initializeIteratorStorage()
{
	void * result = NULL;
	bool initialized = false;

#if __LP64__
	OSCollectionIteratorStorageType storageType = getStorageType();
	switch (storageType) {
	case OSCollectionIteratorStorageUnallocated:
		if (collection->iteratorSize() > sizeof(inlineStorage) || isSubclassed()) {
			collIterator = (void *)kalloc_data(collection->iteratorSize(), Z_WAITOK);
			OSCONTAINER_ACCUMSIZE(collection->iteratorSize());
			if (!collection->initIterator(collIterator)) {
				kfree_data(collIterator, collection->iteratorSize());
				OSCONTAINER_ACCUMSIZE(-((size_t) collection->iteratorSize()));
				collIterator = NULL;
				initialized = false;
				setStorageType(OSCollectionIteratorStorageUnallocated);
			} else {
				setStorageType(OSCollectionIteratorStoragePointer);
				result = collIterator;
				initialized = true;
			}
		} else {
			bzero(&inlineStorage[0], collection->iteratorSize());
			if (!collection->initIterator(&inlineStorage[0])) {
				bzero(&inlineStorage[0], collection->iteratorSize());
				initialized = false;
				setStorageType(OSCollectionIteratorStorageUnallocated);
			} else {
				setStorageType(OSCollectionIteratorStorageInline);
				result = &inlineStorage[0];
				initialized = true;
			}
		}
		break;
	case OSCollectionIteratorStoragePointer:
		// already initialized
		initialized = false;
		break;
	case OSCollectionIteratorStorageInline:
		// already initialized
		initialized = false;
		break;
	default:
		panic("unexpected storage type %u", storageType);
	}
#else
	if (!collIterator) {
		collIterator = (void *)kalloc_data(collection->iteratorSize(), Z_WAITOK);
		OSCONTAINER_ACCUMSIZE(collection->iteratorSize());
		if (!collection->initIterator(collIterator)) {
			kfree_data(collIterator, collection->iteratorSize());
			OSCONTAINER_ACCUMSIZE(-((size_t) collection->iteratorSize()));
			collIterator = NULL;
			initialized = false;
			setStorageType(OSCollectionIteratorStorageUnallocated);
		} else {
			setStorageType(OSCollectionIteratorStoragePointer);
			result = collIterator;
			initialized = true;
		}
	}
#endif /* __LP64__ */

	if (initialized) {
		valid = true;
		initialUpdateStamp = collection->updateStamp;
	}

	return initialized;
}

void *
OSCollectionIterator::getIteratorStorage()
{
	void * result = NULL;

#if __LP64__
	OSCollectionIteratorStorageType storageType = getStorageType();

	switch (storageType) {
	case OSCollectionIteratorStorageUnallocated:
		result = NULL;
		break;
	case OSCollectionIteratorStoragePointer:
		result = collIterator;
		break;
	case OSCollectionIteratorStorageInline:
		result = &inlineStorage[0];
		break;
	default:
		panic("unexpected storage type %u", storageType);
	}
#else
	OSCollectionIteratorStorageType storageType __assert_only = getStorageType();
	assert(storageType == OSCollectionIteratorStoragePointer || storageType == OSCollectionIteratorStorageUnallocated);
	result = collIterator;
#endif /* __LP64__ */

	return result;
}

void
OSCollectionIterator::freeIteratorStorage()
{
#if __LP64__
	OSCollectionIteratorStorageType storageType = getStorageType();

	switch (storageType) {
	case OSCollectionIteratorStorageUnallocated:
		break;
	case OSCollectionIteratorStoragePointer:
		kfree_data(collIterator, collection->iteratorSize());
		OSCONTAINER_ACCUMSIZE(-((size_t) collection->iteratorSize()));
		collIterator = NULL;
		setStorageType(OSCollectionIteratorStorageUnallocated);
		break;
	case OSCollectionIteratorStorageInline:
		bzero(&inlineStorage[0], collection->iteratorSize());
		setStorageType(OSCollectionIteratorStorageUnallocated);
		break;
	default:
		panic("unexpected storage type %u", storageType);
	}
#else
	if (collIterator != NULL) {
		assert(getStorageType() == OSCollectionIteratorStoragePointer);
		kfree_data(collIterator, collection->iteratorSize());
		OSCONTAINER_ACCUMSIZE(-((size_t) collection->iteratorSize()));
		collIterator = NULL;
		setStorageType(OSCollectionIteratorStorageUnallocated);
	} else {
		assert(getStorageType() == OSCollectionIteratorStorageUnallocated);
	}
#endif /* __LP64__ */
}

bool
OSCollectionIterator::isSubclassed()
{
	return getMetaClass() != OSCollectionIterator::metaClass;
}

OSCollectionIteratorStorageType
OSCollectionIterator::getStorageType()
{
#if __LP64__
	// Storage type is in the most significant 2 bits of collIterator
	return (OSCollectionIteratorStorageType)((uintptr_t)(collIterator) >> 62);
#else
	if (collIterator != NULL) {
		return OSCollectionIteratorStoragePointer;
	} else {
		return OSCollectionIteratorStorageUnallocated;
	}
#endif /* __LP64__ */
}

void
OSCollectionIterator::setStorageType(OSCollectionIteratorStorageType storageType)
{
#if __LP64__
	switch (storageType) {
	case OSCollectionIteratorStorageUnallocated:
		if (collIterator != NULL) {
			assert(getStorageType() == OSCollectionIteratorStorageInline);
			collIterator = NULL;
		}
		break;
	case OSCollectionIteratorStoragePointer:
		// Should already be set
		assert(collIterator != NULL);
		assert(getStorageType() == OSCollectionIteratorStoragePointer);
		break;
	case OSCollectionIteratorStorageInline:
		// Set the two most sigificant bits of collIterator to 10b
		collIterator = (void *)(((uintptr_t)collIterator & ~0xC000000000000000) | ((uintptr_t)OSCollectionIteratorStorageInline << 62));
		break;
	default:
		panic("unexpected storage type %u", storageType);
	}
#else
	switch (storageType) {
	case OSCollectionIteratorStorageUnallocated:
		// Should already be set
		assert(collIterator == NULL);
		assert(getStorageType() == OSCollectionIteratorStorageUnallocated);
		break;
	case OSCollectionIteratorStoragePointer:
		// Should already be set
		assert(collIterator != NULL);
		assert(getStorageType() == OSCollectionIteratorStoragePointer);
		break;
	case OSCollectionIteratorStorageInline:
		panic("cannot use inline storage on LP32");
		break;
	default:
		panic("unexpected storage type %u", storageType);
	}
#endif /* __LP64__ */
}

OSObject *
OSCollectionIterator::getNextObject()
{
	OSObject *retObj;
	bool retVal;
	void * storage;

	if (!isValid()) {
		return NULL;
	}

	storage = getIteratorStorage();
	assert(storage != NULL);

	retVal = collection->getNextObjectForIterator(storage, &retObj);
	return (retVal)? retObj : NULL;
}
