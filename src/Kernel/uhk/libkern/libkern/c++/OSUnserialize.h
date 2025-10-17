/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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
/* OSUnserialize.h created by rsulack on Mon 23-Nov-1998 */

#ifndef _OS_OSUNSERIALIZE_H
#define _OS_OSUNSERIALIZE_H

#include <libkern/c++/OSMetaClass.h>
#include <libkern/c++/OSString.h>
#include <libkern/c++/OSPtr.h>

#include <sys/appleapiopts.h>
#include <sys/types.h>

class OSObject;
class OSString;

/*!
 * @header
 *
 * @abstract
 * This header declares the <code>OSUnserializeXML</code> function.
 */


/*!
 * @function OSUnserializeXML
 *
 * @abstract
 * Recreates an OSContainer object
 * from its previously serialized OSContainer class instance data.
 *
 * @param buffer      A buffer containing nul-terminated XML data
 *                    representing the object to be recreated.
 * @param errorString If non-<code>NULL</code>, and the XML parser
 *                    finds an error in <code>buffer</code>,
 *                    <code>*errorString</code> indicates the line number
 *                    and type of error encountered.
 *
 * @result
 * The recreated object, or <code>NULL</code> on failure.
 *
 * @discussion
 * <b>Not safe</b> to call in a primary interrupt handler.
 */
extern "C++" OSPtr<OSObject> OSUnserializeXML(
	const char  * buffer,
	OSString * * errorString = NULL);

extern "C++" OSPtr<OSObject> OSUnserializeXML(
	const char  * buffer,
	OSSharedPtr<OSString>& errorString);

/*!
 * @function OSUnserializeXML
 *
 * @abstract
 * Recreates an OSContainer object
 * from its previously serialized OSContainer class instance data.
 *
 * @param buffer      A buffer containing nul-terminated XML data
 *                    representing the object to be recreated.
 * @param bufferSize  The size of the block of memory. The function
 *                    never scans beyond the first bufferSize bytes.
 * @param errorString If non-<code>NULL</code>, and the XML parser
 *                    finds an error in <code>buffer</code>,
 *                    <code>*errorString</code> indicates the line number
 *                    and type of error encountered.
 *
 * @result
 * The recreated object, or <code>NULL</code> on failure.
 *
 * @discussion
 * <b>Not safe</b> to call in a primary interrupt handler.
 */
extern "C++" OSPtr<OSObject> OSUnserializeXML(
	const char  * buffer,
	size_t        bufferSize,
	OSString *   *errorString = NULL);

extern "C++" OSPtr<OSObject> OSUnserializeXML(
	const char  * buffer,
	size_t        bufferSize,
	OSSharedPtr<OSString> &errorString);

extern "C++" OSPtr<OSObject>
OSUnserializeBinary(const char *buffer, size_t bufferSize, OSString * *errorString);

extern "C++" OSPtr<OSObject>
OSUnserializeBinary(const char *buffer, size_t bufferSize, OSSharedPtr<OSString>& errorString);

#ifdef __APPLE_API_OBSOLETE
extern OSPtr<OSObject> OSUnserialize(const char *buffer, OSString * *errorString = NULL);

extern OSPtr<OSObject> OSUnserialize(const char *buffer, OSSharedPtr<OSString>&  errorString);

#endif /* __APPLE_API_OBSOLETE */

#endif /* _OS_OSUNSERIALIZE_H */
