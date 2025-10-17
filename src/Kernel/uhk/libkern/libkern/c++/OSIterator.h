/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
 * Copyright (c) 1998-1999 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */

#ifndef _OS_OSITERATOR_H
#define _OS_OSITERATOR_H

#include <libkern/c++/OSObject.h>

/*!
 * @header
 *
 * @abstract
 * This header declares the OSIterator collection class.
 */


/*!
 * @class OSIterator
 * @abstract
 * The abstract superclass for Libkern iterators.
 *
 * @discussion
 * OSIterator is the abstract superclass for all Libkern C++ object iterators.
 * It defines the basic interface for iterating and resetting.
 * See @link //apple_ref/cpp/macro/OSCollection OSCollection@/link and
 * @link //apple_ref/cpp/macro/OSCollectionIterator OSCollectionIterator@/link
 * for more information.
 *
 * With very few exceptions in the I/O Kit, all Libkern-based C++
 * classes, functions, and macros are <b>unsafe</b>
 * to use in a primary interrupt context.
 * Consult the I/O Kit documentation related to primary interrupts
 * for more information.
 *
 * OSIterator provides no concurrency protection.
 */
class OSIterator : public OSObject
{
	OSDeclareAbstractStructors(OSIterator);

public:
/*!
 * @function reset
 *
 * @abstract
 * Resets the iterator to the beginning of the collection,
 * as if it had just been created.
 *
 * @discussion
 * Subclasses must implement this pure virtual member function.
 */
	virtual void reset() = 0;


/*!
 * @function isValid
 *
 * @abstract
 * Check that the collection hasn't been modified during iteration.
 *
 * @result
 * <code>true</code> if the iterator is valid for continued use,
 * <code>false</code> otherwise
 * (typically because the collection being iterated has been modified).
 *
 * @discussion
 * Subclasses must implement this pure virtual member function.
 */
	virtual bool isValid() = 0;


/*!
 * @function getNextObject
 *
 * @abstract
 * Advances to and returns the next object in the iteration.
 *
 * @return
 * The next object in the iteration context,
 * <code>NULL</code> if there is no next object
 * or if the iterator is no longer valid.
 *
 * @discussion
 * The returned object will be released if removed from the collection;
 * if you plan to store the reference, you should call
 * <code>@link
 * //apple_ref/cpp/instm/OSObject/retain/virtualvoid/()
 * retain@/link</code>
 * on that object.
 *
 * Subclasses must implement this pure virtual function
 * to check for validity with
 * <code>@link isValid isValid@/link</code>,
 * and then to advance the iteration context to the next object (if any)
 * and return that next object, or <code>NULL</code> if there is none.
 */
	virtual OSObject *getNextObject() = 0;

	OSMetaClassDeclareReservedUnused(OSIterator, 0);
	OSMetaClassDeclareReservedUnused(OSIterator, 1);
	OSMetaClassDeclareReservedUnused(OSIterator, 2);
	OSMetaClassDeclareReservedUnused(OSIterator, 3);
};

#endif /* ! _OS_OSITERATOR_H */
