/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
#ifndef BinaryPropertyList_h
#define BinaryPropertyList_h

#include <CoreFoundation/CoreFoundation.h>

#include <wtf/Forward.h>
#include <wtf/Vector.h>

// Writes a limited subset of binary property lists.
// Covers only what's needed for writing browser history as of this writing.
class BinaryPropertyListObjectStream {
public:
    // Call writeBooleanTrue to write the boolean true value.
    // A single shared object will be used in the serialized list.
    virtual void writeBooleanTrue() = 0;

    // Call writeInteger to write an integer value.
    // A single shared object will be used for each integer in the serialized list.
    virtual void writeInteger(int) = 0;

    // Call writeString to write a string value.
    // A single shared object will be used for each string in the serialized list.
    virtual void writeString(const String&) = 0;

    // Call writeUniqueString instead of writeString when it's unlikely the
    // string will be written twice in the same property list; this saves hash
    // table overhead for such strings. A separate object will be used for each
    // of these strings in the serialized list.
    virtual void writeUniqueString(const String&) = 0;
    virtual void writeUniqueString(const char*) = 0;

    // Call writeIntegerArray instead of writeArrayStart/writeArrayEnd for
    // arrays entirely composed of integers. A single shared object will be used
    // for each identical array in the serialized list. Warning: The integer
    // pointer must remain valid until the writeBinaryPropertyList function
    // returns, because these lists are put into a hash table without copying
    // them -- that's OK if the client already has a Vector<int>.
    virtual void writeIntegerArray(const int*, size_t) = 0;

    // After calling writeArrayStart, write array elements.
    // Then call writeArrayEnd, passing in the result from writeArrayStart.
    // A separate object will be used for each of these arrays in the serialized list.
    virtual size_t writeArrayStart() = 0;
    virtual void writeArrayEnd(size_t resultFromWriteArrayStart) = 0;

    // After calling writeDictionaryStart, write all keys, then all values.
    // Then call writeDictionaryEnd, passing in the result from writeDictionaryStart.
    // A separate object will be used for each dictionary in the serialized list.
    virtual size_t writeDictionaryStart() = 0;
    virtual void writeDictionaryEnd(size_t resultFromWriteDictionaryStart) = 0;

protected:
    virtual ~BinaryPropertyListObjectStream() { }
};

class BinaryPropertyListWriter {
public:
    // Calls writeObjects once to prepare for writing and determine how big a
    // buffer is required. Then calls buffer to get the appropriately-sized
    // buffer, then calls writeObjects a second time and writes the property list.
    void writePropertyList();

protected:
    virtual ~BinaryPropertyListWriter() { }

private:
    // Called by writePropertyList.
    // Must call the object stream functions for the objects to be written
    // into the property list.
    virtual void writeObjects(BinaryPropertyListObjectStream&) = 0;

    // Called by writePropertyList.
    // Returns the buffer that the writer should write into.
    virtual UInt8* buffer(size_t) = 0;

    friend class BinaryPropertyListPlan;
    friend class BinaryPropertyListSerializer;
};

#endif // BinaryPropertyList_h
