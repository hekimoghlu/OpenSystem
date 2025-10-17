/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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

//===-- DOTGraphTraitsPass.h - Print/View dotty graphs-----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Templates to create dotty viewer and printer passes for GraphTraits graphs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOT_GRAPHTRAITS_PASS_H
#define LLVM_ANALYSIS_DOT_GRAPHTRAITS_PASS_H

#include "llvm/Pass.h"
#include "llvm/Analysis/CFGPrinter.h"

namespace llvm {
template <class Analysis, bool Simple>
struct DOTGraphTraitsViewer : public FunctionPass {
  std::string Name;

  DOTGraphTraitsViewer(std::string GraphName, char &ID) : FunctionPass(ID) {
    Name = GraphName;
  }

  virtual bool runOnFunction(Function &F) {
    Analysis *Graph;
    std::string Title, GraphName;
    Graph = &getAnalysis<Analysis>();
    GraphName = DOTGraphTraits<Analysis*>::getGraphName(Graph);
    Title = GraphName + " for '" + F.getName().str() + "' function";
    ViewGraph(Graph, Name, Simple, Title);

    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<Analysis>();
  }
};

template <class Analysis, bool Simple>
struct DOTGraphTraitsPrinter : public FunctionPass {

  std::string Name;

  DOTGraphTraitsPrinter(std::string GraphName, char &ID)
    : FunctionPass(ID) {
    Name = GraphName;
  }

  virtual bool runOnFunction(Function &F) {
    Analysis *Graph;
    std::string Filename = Name + "." + F.getName().str() + ".dot";
    errs() << "Writing '" << Filename << "'...";

    std::string ErrorInfo;
    raw_fd_ostream File(Filename.c_str(), ErrorInfo);
    Graph = &getAnalysis<Analysis>();

    std::string Title, GraphName;
    GraphName = DOTGraphTraits<Analysis*>::getGraphName(Graph);
    Title = GraphName + " for '" + F.getName().str() + "' function";

    if (ErrorInfo.empty())
      WriteGraph(File, Graph, Simple, Title);
    else
      errs() << "  error opening file for writing!";
    errs() << "\n";
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<Analysis>();
  }
};
}
#endif
