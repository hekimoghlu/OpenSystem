/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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

//===- WorkList.cpp - Analyzer work-list implementation--------------------===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
// Defines different worklist implementations for the static analyzer.
//
//===----------------------------------------------------------------------===//

#include "language/Core/StaticAnalyzer/Core/PathSensitive/WorkList.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/EntryPointStats.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/DenseSet.h"
#include "toolchain/ADT/PriorityQueue.h"
#include "toolchain/ADT/STLExtras.h"
#include <deque>
#include <vector>

using namespace language::Core;
using namespace ento;

#define DEBUG_TYPE "WorkList"

STAT_MAX(MaxQueueSize, "Maximum size of the worklist");
STAT_MAX(MaxReachableSize, "Maximum size of auxiliary worklist set");

//===----------------------------------------------------------------------===//
// Worklist classes for exploration of reachable states.
//===----------------------------------------------------------------------===//

namespace {

class DFS : public WorkList {
  SmallVector<WorkListUnit, 20> Stack;

public:
  bool hasWork() const override {
    return !Stack.empty();
  }

  void enqueue(const WorkListUnit& U) override {
    Stack.push_back(U);
  }

  WorkListUnit dequeue() override {
    assert(!Stack.empty());
    const WorkListUnit& U = Stack.back();
    Stack.pop_back(); // This technically "invalidates" U, but we are fine.
    return U;
  }
};

class BFS : public WorkList {
  std::deque<WorkListUnit> Queue;

public:
  bool hasWork() const override {
    return !Queue.empty();
  }

  void enqueue(const WorkListUnit& U) override {
    Queue.push_back(U);
  }

  WorkListUnit dequeue() override {
    WorkListUnit U = Queue.front();
    Queue.pop_front();
    return U;
  }
};

} // namespace

// Place the dstor for WorkList here because it contains virtual member
// functions, and we the code for the dstor generated in one compilation unit.
WorkList::~WorkList() = default;

std::unique_ptr<WorkList> WorkList::makeDFS() {
  return std::make_unique<DFS>();
}

std::unique_ptr<WorkList> WorkList::makeBFS() {
  return std::make_unique<BFS>();
}

namespace {

  class BFSBlockDFSContents : public WorkList {
    std::deque<WorkListUnit> Queue;
    SmallVector<WorkListUnit, 20> Stack;

  public:
    bool hasWork() const override {
      return !Queue.empty() || !Stack.empty();
    }

    void enqueue(const WorkListUnit& U) override {
      if (U.getNode()->getLocation().getAs<BlockEntrance>())
        Queue.push_front(U);
      else
        Stack.push_back(U);
    }

    WorkListUnit dequeue() override {
      // Process all basic blocks to completion.
      if (!Stack.empty()) {
        const WorkListUnit& U = Stack.back();
        Stack.pop_back(); // This technically "invalidates" U, but we are fine.
        return U;
      }

      assert(!Queue.empty());
      // Don't use const reference.  The subsequent pop_back() might make it
      // unsafe.
      WorkListUnit U = Queue.front();
      Queue.pop_front();
      return U;
    }
  };

} // namespace

std::unique_ptr<WorkList> WorkList::makeBFSBlockDFSContents() {
  return std::make_unique<BFSBlockDFSContents>();
}

namespace {

class UnexploredFirstStack : public WorkList {
  /// Stack of nodes known to have statements we have not traversed yet.
  SmallVector<WorkListUnit, 20> StackUnexplored;

  /// Stack of all other nodes.
  SmallVector<WorkListUnit, 20> StackOthers;

  using BlockID = unsigned;
  using LocIdentifier = std::pair<BlockID, const StackFrameContext *>;

  toolchain::DenseSet<LocIdentifier> Reachable;

public:
  bool hasWork() const override {
    return !(StackUnexplored.empty() && StackOthers.empty());
  }

  void enqueue(const WorkListUnit &U) override {
    const ExplodedNode *N = U.getNode();
    auto BE = N->getLocation().getAs<BlockEntrance>();

    if (!BE) {
      // Assume the choice of the order of the preceding block entrance was
      // correct.
      StackUnexplored.push_back(U);
    } else {
      LocIdentifier LocId = std::make_pair(
          BE->getBlock()->getBlockID(),
          N->getLocationContext()->getStackFrame());
      auto InsertInfo = Reachable.insert(LocId);

      if (InsertInfo.second) {
        StackUnexplored.push_back(U);
      } else {
        StackOthers.push_back(U);
      }
    }
    MaxReachableSize.updateMax(Reachable.size());
    MaxQueueSize.updateMax(StackUnexplored.size() + StackOthers.size());
  }

  WorkListUnit dequeue() override {
    if (!StackUnexplored.empty()) {
      WorkListUnit &U = StackUnexplored.back();
      StackUnexplored.pop_back();
      return U;
    } else {
      WorkListUnit &U = StackOthers.back();
      StackOthers.pop_back();
      return U;
    }
  }
};

} // namespace

std::unique_ptr<WorkList> WorkList::makeUnexploredFirst() {
  return std::make_unique<UnexploredFirstStack>();
}

namespace {
class UnexploredFirstPriorityQueue : public WorkList {
  using BlockID = unsigned;
  using LocIdentifier = std::pair<BlockID, const StackFrameContext *>;

  // How many times each location was visited.
  // Is signed because we negate it later in order to have a reversed
  // comparison.
  using VisitedTimesMap = toolchain::DenseMap<LocIdentifier, int>;

  // Compare by number of times the location was visited first (negated
  // to prefer less often visited locations), then by insertion time (prefer
  // expanding nodes inserted sooner first).
  using QueuePriority = std::pair<int, unsigned long>;
  using QueueItem = std::pair<WorkListUnit, QueuePriority>;

  // Number of inserted nodes, used to emulate DFS ordering in the priority
  // queue when insertions are equal.
  unsigned long Counter = 0;

  // Number of times a current location was reached.
  VisitedTimesMap NumReached;

  // The top item is the largest one.
  toolchain::PriorityQueue<QueueItem, std::vector<QueueItem>, toolchain::less_second>
      queue;

public:
  bool hasWork() const override {
    return !queue.empty();
  }

  void enqueue(const WorkListUnit &U) override {
    const ExplodedNode *N = U.getNode();
    unsigned NumVisited = 0;
    if (auto BE = N->getLocation().getAs<BlockEntrance>()) {
      LocIdentifier LocId = std::make_pair(
          BE->getBlock()->getBlockID(),
          N->getLocationContext()->getStackFrame());
      NumVisited = NumReached[LocId]++;
    }

    queue.push(std::make_pair(U, std::make_pair(-NumVisited, ++Counter)));
  }

  WorkListUnit dequeue() override {
    QueueItem U = queue.top();
    queue.pop();
    return U.first;
  }
};
} // namespace

std::unique_ptr<WorkList> WorkList::makeUnexploredFirstPriorityQueue() {
  return std::make_unique<UnexploredFirstPriorityQueue>();
}

namespace {
class UnexploredFirstPriorityLocationQueue : public WorkList {
  using LocIdentifier = const CFGBlock *;

  // How many times each location was visited.
  // Is signed because we negate it later in order to have a reversed
  // comparison.
  using VisitedTimesMap = toolchain::DenseMap<LocIdentifier, int>;

  // Compare by number of times the location was visited first (negated
  // to prefer less often visited locations), then by insertion time (prefer
  // expanding nodes inserted sooner first).
  using QueuePriority = std::pair<int, unsigned long>;
  using QueueItem = std::pair<WorkListUnit, QueuePriority>;

  // Number of inserted nodes, used to emulate DFS ordering in the priority
  // queue when insertions are equal.
  unsigned long Counter = 0;

  // Number of times a current location was reached.
  VisitedTimesMap NumReached;

  // The top item is the largest one.
  toolchain::PriorityQueue<QueueItem, std::vector<QueueItem>, toolchain::less_second>
      queue;

public:
  bool hasWork() const override {
    return !queue.empty();
  }

  void enqueue(const WorkListUnit &U) override {
    const ExplodedNode *N = U.getNode();
    unsigned NumVisited = 0;
    if (auto BE = N->getLocation().getAs<BlockEntrance>())
      NumVisited = NumReached[BE->getBlock()]++;

    queue.push(std::make_pair(U, std::make_pair(-NumVisited, ++Counter)));
  }

  WorkListUnit dequeue() override {
    QueueItem U = queue.top();
    queue.pop();
    return U.first;
  }

};

}

std::unique_ptr<WorkList> WorkList::makeUnexploredFirstPriorityLocationQueue() {
  return std::make_unique<UnexploredFirstPriorityLocationQueue>();
}
