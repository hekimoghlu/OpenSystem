/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
// Plugin interface for internal Security plug-ins
//
#ifndef _H_ANTLRPLUGIN
#define _H_ANTLRPLUGIN

#include <Security/CodeSigning.h>
#include "requirement.h"
#include <cstdio>
#include <string>

namespace Security {
namespace CodeSigning {


//
// The plugin proxy.
//
// During loading, one instance of this object will be created by the plugin
// and returned through the (one and only) dynamically-linked method of the plugin.
// All further interaction then proceeds through methods of this object.
//
//
class AntlrPlugin {
public:
	typedef const Requirement *FileRequirement(std::FILE *source, std::string &errors);
	FileRequirement *fileRequirement;
	typedef const Requirements *FileRequirements(std::FILE *source, std::string &errors);
	FileRequirements *fileRequirements;
	typedef const BlobCore *FileGeneric(std::FILE *source, std::string &errors);
	FileGeneric *fileGeneric;
	typedef const Requirement *StringRequirement(std::string source, std::string &errors);
	StringRequirement *stringRequirement;
	typedef const Requirements *StringRequirements(std::string source, std::string &errors);
	StringRequirements *stringRequirements;
	typedef const BlobCore *StringGeneric(std::string source, std::string &errors);
	StringGeneric *stringGeneric;
};

extern "C" {
	AntlrPlugin *findAntlrPlugin();
	typedef AntlrPlugin *FindAntlrPlugin();
}

#define FINDANTLRPLUGIN "findAntlrPlugin"


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_ANTLRPLUGIN
