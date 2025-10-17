/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
// cssmcred - enhanced PodWrappers and construction aids for ACL credentials
//
#include <security_cdsa_utilities/cssmcred.h>


namespace Security {

//
// Scan a SampleGroup for samples with a given CSSM_SAMPLE_TYPE.
// Collect all matching samples into a list (which is cleared to begin with).
// Return true if any were found, false if none.
// Throw if any of the samples are obviously malformed.
//
bool SampleGroup::collect(CSSM_SAMPLE_TYPE sampleType, list<CssmSample> &matches) const
{
	for (uint32 n = 0; n < length(); n++) {
		TypedList sample = (*this)[n];
		sample.checkProper();
		if (sample.type() == sampleType) {
			sample.snip();	// skip sample type
			matches.push_back(sample);
		}
	}
	return !matches.empty();
}


//
// AccessCredentials
//
const AccessCredentials& AccessCredentials::null_credential()
{
    static const CSSM_ACCESS_CREDENTIALS null_credentials = { "" };    // and more nulls
    return AccessCredentials::overlay(null_credentials);
}

void AccessCredentials::tag(const char *tagString)
{
	if (tagString == NULL)
		EntryTag[0] = '\0';
	else if (strlen(tagString) > CSSM_MODULE_STRING_SIZE)
		CssmError::throwMe(CSSM_ERRCODE_INVALID_ACL_ENTRY_TAG);
	else
		strcpy(EntryTag, tagString);
}

bool AccessCredentials::authorizesUI() const {
    list<CssmSample> uisamples;

    if(samples().collect(CSSM_SAMPLE_TYPE_KEYCHAIN_PROMPT, uisamples)) {
        // The existence of a lone keychain prompt gives UI access
        return true;
    }

    samples().collect(CSSM_SAMPLE_TYPE_KEYCHAIN_LOCK, uisamples);
    samples().collect(CSSM_SAMPLE_TYPE_THRESHOLD, uisamples);

    for (list<CssmSample>::iterator it = uisamples.begin(); it != uisamples.end(); it++) {
        TypedList &sample = *it;

        if(!sample.isProper()) {
            secnotice("integrity", "found a non-proper sample, skipping...");
            continue;
        }

        switch (sample.type()) {
            case CSSM_SAMPLE_TYPE_KEYCHAIN_PROMPT:
                // these credentials allow UI
                return true;
        }
    }

    // no interesting credential found; no UI for you
    return false;
}

//
// AutoCredentials self-constructing credentials structure
//
AutoCredentials::AutoCredentials(Allocator &alloc) : allocator(alloc)
{
	init();
}

AutoCredentials::AutoCredentials(Allocator &alloc, uint32 nSamples) : allocator(alloc)
{
	init();
	getSample(nSamples - 1);	// extend array to nSamples elements
}

void AutoCredentials::init()
{
	sampleArray = NULL;
	nSamples = 0;
}


CssmSample &AutoCredentials::getSample(uint32 n)
{
	if (n >= nSamples) {
		sampleArray = allocator.alloc<CssmSample>(sampleArray, nSamples = n + 1);
		Samples.Samples = sampleArray;
		Samples.NumberOfSamples = nSamples;
	}
	return sampleArray[n];
}

}	// end namespace Security
