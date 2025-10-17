/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#ifndef	_OID_PARSER_H_
#define _OID_PARSER_H_

#include <Security/cssmtype.h>

/*
 * Generated strings go into a client-allocated char array of 
 * this size.
 */
#define OID_PARSER_STRING_SIZE	120

class OidParser
{
private:
	CSSM_DATA_PTR		configData;		// contents of  dumpasn1.cfg
public:
	/* costruct with noConfig true - skip reading config file */
	OidParser(bool noConfig=false);
	~OidParser();

	/*
	 * Parse an Intel-style OID, generating a C string in 
	 * caller-supplied buffer.
	 */
	void oidParse(
		const unsigned char	*oidp,
		unsigned			oidLen,
		char				*strBuf,
		size_t				strLen);

};

#endif	/* _OID_PARSER_H_ */
