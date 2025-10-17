/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
// reqparser - interface to Requirement language parser/compiler
//
#include "reqparser.h"
#include "antlrplugin.h"
#include "cserror.h"
#include "codesigning_dtrace.h"
#include <CoreFoundation/CoreFoundation.h>
#include <security_utilities/osxcode.h>
#include <security_utilities/logging.h>
#include <Security/SecFramework.h>

namespace Security {
namespace CodeSigning {


struct PluginHost {
	PluginHost();
	RefPointer<LoadableBundle> plugin;
	AntlrPlugin *antlr;
};

ModuleNexus<PluginHost> plugin;


//
// The PluginHost constructor runs under the protection of ModuleNexus's constructor,
// so it doesn't have to worry about thread safety and such.
//
PluginHost::PluginHost()
{
	if (CFBundleRef securityFramework = SecFrameworkGetBundle())
		if (CFRef<CFURLRef> plugins = CFBundleCopyBuiltInPlugInsURL(securityFramework))
			if (CFRef<CFURLRef> pluginURL = makeCFURL("csparser.bundle", true, plugins)) {
				plugin = new LoadableBundle(cfString(pluginURL).c_str());
				plugin->load();
				CODESIGN_LOAD_ANTLR();
				antlr = reinterpret_cast<FindAntlrPlugin *>(plugin->lookupSymbol(FINDANTLRPLUGIN))();
				return;
			}
				
	// can't load plugin - fail
    Syslog::warning("code signing problem: unable to load csparser plug-in");
	MacOSError::throwMe(errSecCSInternalError);
}


//
// Drive a parsing function through the plugin harness and translate any errors
// into a CFError exception.
//
template <class Result, class Source>
const Result *parse(Source source, const Result *(*AntlrPlugin::*func)(Source, string &))
{
	string errors;
	if (const Result *result = (plugin().antlr->*func)(source, errors))
		return result;
	else
		CSError::throwMe(errSecCSReqInvalid, kSecCFErrorRequirementSyntax, CFTempString(errors));		
}


//
// Implement the template instances by passing them through the plugin's eye-of-the-needle.
// Any other combination of input and output types will cause linker errors.
//
template <>
const Requirement *RequirementParser<Requirement>::operator () (std::FILE *source)
{
	return parse(source, &AntlrPlugin::fileRequirement);
}

template <>
const Requirement *RequirementParser<Requirement>::operator () (const std::string &source)
{
	return parse(source, &AntlrPlugin::stringRequirement);
}

template <>
const Requirements *RequirementParser<Requirements>::operator () (std::FILE *source)
{
	return parse(source, &AntlrPlugin::fileRequirements);
}

template <>
const Requirements *RequirementParser<Requirements>::operator () (const std::string &source)
{
	return parse(source, &AntlrPlugin::stringRequirements);
}

template <>
const BlobCore *RequirementParser<BlobCore>::operator () (std::FILE *source)
{
	return parse(source, &AntlrPlugin::fileGeneric);
}

template <>
const BlobCore *RequirementParser<BlobCore>::operator () (const std::string &source)
{
	return parse(source, &AntlrPlugin::stringGeneric);
}


}	// CodeSigning
}	// Security
