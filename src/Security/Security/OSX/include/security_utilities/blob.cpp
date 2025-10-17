/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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
//
// blob - generic extensible binary blob frame
//
#include "blob.h"
#include <security_utilities/unix++.h>

namespace Security {


//
// Content access and validation calls
//
char *BlobCore::stringAt(Offset offset)
{
	char *s = at<char>(offset);
	if (offset < this->length() && memchr(s, 0, this->length() - offset))
		return s;
	else
		return NULL;
}

const char *BlobCore::stringAt(Offset offset) const
{
	const char *s = at<const char>(offset);
	if (offset < this->length() && memchr(s, 0, this->length() - offset))
		return s;
	else
		return NULL;
}


//
// Read a blob from a standard file stream.
// Reads in one pass, so it's suitable for transmission over pipes and networks.
// The blob is allocated with malloc(3).
// On error, sets errno and returns NULL; in which case the input stream may
// be partially consumed.
//
BlobCore *BlobCore::readBlob(int fd, size_t offset, uint32_t magic, size_t minSize, size_t maxSize)
{
	BlobCore header;
	if (::pread(fd, &header, sizeof(header), offset) == sizeof(header))
		if (header.validateBlob(magic, minSize, maxSize))
			if (BlobCore *blob = (BlobCore *)malloc(header.length())) {
				memcpy(blob, &header, sizeof(header));
				size_t remainder = header.length() - sizeof(header);
				if (::pread(fd, blob+1, remainder, offset + sizeof(header)) == ssize_t(remainder))
					return blob;
				free(blob);
				errno = EINVAL;
			}
	return NULL;
}

BlobCore *BlobCore::readBlob(int fd, uint32_t magic, size_t minSize, size_t maxSize)
{
	BlobCore header;
	if (::read(fd, &header, sizeof(header)) == sizeof(header))
		if (header.validateBlob(magic, minSize, maxSize))
			if (BlobCore *blob = (BlobCore *)malloc(header.length())) {
				memcpy(blob, &header, sizeof(header));
				size_t remainder = header.length() - sizeof(header);
				if (::read(fd, blob+1, remainder) == ssize_t(remainder))
					return blob;
				free(blob);
				errno = EINVAL;
			}
	return NULL;
}

BlobCore *BlobCore::readBlob(std::FILE *file, uint32_t magic, size_t minSize, size_t maxSize)
{
	BlobCore header;
	if (::fread(&header, sizeof(header), 1, file) == 1)
		if (header.validateBlob(magic, minSize, maxSize))
			if (BlobCore *blob = (BlobCore *)malloc(header.length())) {
				memcpy(blob, &header, sizeof(header));
				if (::fread(blob+1, header.length() - sizeof(header), 1, file) == 1)
					return blob;
				free(blob);
				errno = EINVAL;
			}
	return NULL;
}


//
// BlobWrappers
//
BlobWrapper *BlobWrapper::alloc(size_t length, Magic magic /* = _magic */)
{
	size_t wrapLength = length + sizeof(BlobCore);
	if (wrapLength < length)	// overflow
		return NULL;
	BlobWrapper *w = (BlobWrapper *)malloc(wrapLength);
	if (w)
		w->BlobCore::initialize(magic, wrapLength);
	return w;
}

BlobWrapper *BlobWrapper::alloc(const void *data, size_t length, Magic magic /* = _magic */)
{
	BlobWrapper *w = alloc(length, magic);
	if (w)
		memcpy(w->data(), data, w->length());
	return w;
}


}	// Security
