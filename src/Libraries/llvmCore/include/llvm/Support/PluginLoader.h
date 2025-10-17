/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 9, 2022.
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

//===-- llvm/Support/PluginLoader.h - Plugin Loader for Tools ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A tool can #include this file to get a -load option that allows the user to
// load arbitrary shared objects into the tool's address space.  Note that this
// header can only be included by a program ONCE, so it should never to used by
// library authors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PLUGINLOADER_H
#define LLVM_SUPPORT_PLUGINLOADER_H

#include "llvm/Support/CommandLine.h"

namespace llvm {
  struct PluginLoader {
    void operator=(const std::string &Filename);
    static unsigned getNumPlugins();
    static std::string& getPlugin(unsigned num);
  };

#ifndef DONT_GET_PLUGIN_LOADER_OPTION
  // This causes operator= above to be invoked for every -load option.
  static cl::opt<PluginLoader, false, cl::parser<std::string> >
    LoadOpt("load", cl::ZeroOrMore, cl::value_desc("pluginfilename"),
            cl::desc("Load the specified plugin"));
#endif
}

#endif
