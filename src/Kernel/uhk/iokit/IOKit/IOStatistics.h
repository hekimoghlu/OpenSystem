/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#ifndef _IOKIT_STATISTICS_H
#define _IOKIT_STATISTICS_H

#define IOSTATISTICS_SIG            'IOST'
#define IOSTATISTICS_SIG_USERCLIENT 'IOSU'
#define IOSTATISTICS_SIG_WORKLOOP   'IOSW'

/* Update when the binary format changes */
#define IOSTATISTICS_VER                        0x2

enum {
	kIOStatisticsDriverNameLength  = 64,
	kIOStatisticsClassNameLength   = 64,
	kIOStatisticsProcessNameLength = 20
};

enum {
	kIOStatisticsDerivedEventSourceCounter = 0,
	kIOStatisticsTimerEventSourceCounter,
	kIOStatisticsCommandGateCounter,
	kIOStatisticsCommandQueueCounter,
	kIOStatisticsInterruptEventSourceCounter,
	kIOStatisticsFilterInterruptEventSourceCounter
};

typedef uint32_t IOStatisticsCounterType;

enum {
	kIOStatisticsGeneral = 0,
	kIOStatisticsWorkLoop,
	kIOStatisticsUserClient
};

/* Keep our alignments as intended */

#pragma pack(4)

/* Event Counters */

typedef struct IOStatisticsInterruptEventSources {
	uint32_t created;
	uint32_t produced;
	uint32_t checksForWork;
} IOStatisticsInterruptEventSources;

typedef struct IOStatisticsTimerEventSources {
	uint32_t created;
	uint32_t openGateCalls;
	uint32_t closeGateCalls;
	uint64_t timeOnGate;
	uint32_t timeouts;
	uint32_t checksForWork;
} IOStatisticsTimerEventSources;

typedef struct IOStatisticsDerivedEventSources {
	uint32_t created;
	uint32_t openGateCalls;
	uint32_t closeGateCalls;
	uint64_t timeOnGate;
} IOStatisticsDerivedEventSources;

typedef struct IOStatisticsCommandGates {
	uint32_t created;
	uint32_t openGateCalls;
	uint32_t closeGateCalls;
	uint64_t timeOnGate;
	uint32_t actionCalls;
} IOStatisticsCommandGates;

typedef struct IOStatisticsCommandQueues {
	uint32_t created;
	uint32_t actionCalls;
} IOStatisticsCommandQueues;

typedef struct IOStatisticsUserClients {
	uint32_t created;
	uint32_t clientCalls;
} IOStatisticsUserClients;

/* General mode */

typedef struct IOStatisticsHeader {
	uint32_t sig; /* 'IOST' */
	uint32_t ver; /* incremented with every data revision */

	uint32_t seq; /* sequence ID */

	uint32_t globalStatsOffset;
	uint32_t kextStatsOffset;
	uint32_t memoryStatsOffset;
	uint32_t classStatsOffset;
	uint32_t counterStatsOffset;
	uint32_t kextIdentifiersOffset;
	uint32_t classNamesOffset;

	/* struct IOStatisticsGlobal */
	/* struct IOStatisticsKext */
	/* struct IOStatisticsMemory */
	/* struct IOStatisticsClass */
	/* struct IOStatisticsCounter */
	/* struct IOStatisticsKextIdentifier */
	/* struct IOStatisticsClassName */
} IOStatisticsHeader;

typedef struct IOStatisticsGlobal {
	uint32_t kextCount;
	uint32_t classCount;
	uint32_t workloops;
} IOStatisticsGlobal;

typedef struct IOStatisticsKext {
	uint32_t loadTag;
	uint32_t loadSize;
	uint32_t wiredSize;
	uint32_t classes; /* Number of classes owned */
	uint32_t classIndexes[]; /* Variable length array of owned class indexes */
} IOStatisticsKext;

typedef struct IOStatisticsMemory {
	uint32_t allocatedSize;
	uint32_t freedSize;
	uint32_t allocatedAlignedSize;
	uint32_t freedAlignedSize;
	uint32_t allocatedContiguousSize;
	uint32_t freedContiguousSize;
	uint32_t allocatedPageableSize;
	uint32_t freedPageableSize;
} IOStatisticsMemory;

typedef struct IOStatisticsClass {
	uint32_t classID;
	uint32_t superClassID;
	uint32_t classSize;
} IOStatisticsClass;

typedef struct IOStatisticsCounter {
	uint32_t classID;
	uint32_t classInstanceCount;
	struct IOStatisticsUserClients userClientStatistics;
	struct IOStatisticsInterruptEventSources interruptEventSourceStatistics;
	struct IOStatisticsInterruptEventSources filterInterruptEventSourceStatistics;
	struct IOStatisticsTimerEventSources timerEventSourceStatistics;
	struct IOStatisticsCommandGates commandGateStatistics;
	struct IOStatisticsCommandQueues commandQueueStatistics;
	struct IOStatisticsDerivedEventSources derivedEventSourceStatistics;
} IOStatisticsCounter;

typedef struct IOStatisticsKextIdentifier {
	char identifier[kIOStatisticsDriverNameLength];
} IOStatisticsKextIdentifier;

typedef struct IOStatisticsClassName {
	char name[kIOStatisticsClassNameLength];
} IOStatisticsClassName;

/* WorkLoop mode */

typedef struct IOStatisticsWorkLoop {
	uint32_t attachedEventSources;
	uint64_t timeOnGate;
	uint32_t kextLoadTag;
	uint32_t dependentKexts;
	uint32_t dependentKextLoadTags[];
} IOStatisticsWorkLoop;

typedef struct IOStatisticsWorkLoopHeader {
	uint32_t sig; /* 'IOSW */
	uint32_t ver; /* incremented with every data revision */
	uint32_t seq; /* sequence ID */
	uint32_t workloopCount;
	struct IOStatisticsWorkLoop workLoopStats;
} IOStatisticsWorkLoopHeader;

/* UserClient mode */

typedef struct IOStatisticsUserClientCall {
	char processName[kIOStatisticsProcessNameLength];
	int32_t pid;
	uint32_t calls;
} IOStatisticsUserClientCall;

typedef struct IOStatisticsUserClientHeader {
	uint32_t sig; /* 'IOSU */
	uint32_t ver; /* incremented with every data revision */
	uint32_t seq; /* sequence ID */
	uint32_t processes;
	struct IOStatisticsUserClientCall userClientCalls[];
} IOStatisticsUserClientHeader;

#pragma pack()

#endif /* _IOKIT_STATISTICS_H */
