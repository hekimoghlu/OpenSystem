/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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
// drmaker - create Designated Requirements
//
#ifndef _H_DRMAKER
#define _H_DRMAKER

#include "reqmaker.h"
#include <Security/SecAsn1Types.h>

namespace Security {
namespace CodeSigning {


//
// Some useful certificate OID markers
//
extern const CSSM_DATA adcSdkMarkerOID;
extern const CSSM_DATA devIdSdkMarkerOID;
extern const CSSM_DATA devIdLeafMarkerOID;



//
// A Maker of Designated Requirements
//
class DRMaker : public Requirement::Maker {
public:
	DRMaker(const Requirement::Context &context);
	virtual ~DRMaker();
	
	const Requirement::Context &ctx;
	
public:
	Requirement *make();

private:
	void appleAnchor();
	void nonAppleAnchor();
	bool isIOSSignature();
	bool isDeveloperIDSignature();
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_DRMAKER
