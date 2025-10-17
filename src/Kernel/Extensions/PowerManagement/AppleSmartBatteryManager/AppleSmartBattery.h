/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
#ifndef __AppleSmartBattery__
#define __AppleSmartBattery__

#include <AppleFeatures/AppleFeatures.h>
#include <IOKit/IOService.h>
#include <IOKit/pwr_mgt/IOPMPowerSource.h>
#include <IOKit/IOReporter.h>

#define TARGET_OS_OSX_X86   (TARGET_OS_OSX && !TARGET_CPU_ARM64)    // Non-Apple Silicon Mac platforms
#define TARGET_OS_OSX_AS    (TARGET_OS_OSX && TARGET_CPU_ARM64)     // Apple Silicon Mac platforms

#if TARGET_OS_OSX_X86
#include <IOKit/smbus/IOSMBusController.h>
#include <IOKit/acpi/IOACPIPlatformDevice.h>
#endif


#define kBatteryPollingDebugKey     "BatteryPollingPeriodOverride"

class AppleSmartBatteryManager;
class AppleSmartBatteryHFDataClient;

struct CommandStruct_s;
typedef CommandStruct_s CommandStruct;

typedef struct {
    CommandStruct   *table;
    int             count;
} CommandTable;

enum {
    kUseLastPath    = 0,
    kBoot           = 1,
    kFull           = 2,
    kUserVis        = 4,
};

typedef struct {
    const OSSymbol    *regKey;
    SMCKey      key;
    int32_t     byteCnt;
    int pathBits;
    enum asbUcKeyType keyType;
} smcToRegistry;

typedef struct {
    AbsoluteTime absTime;
    OSDictionary *dict;
} streamTimerEvent;

typedef struct {
    IOSharedDataQueue *queue;
    AppleSmartBatteryHFDataClient *client;
    AbsoluteTime periodTime;
    AbsoluteTime wakeupTime;
} streamQueueCollect;

typedef void (*hfdata_callback_t) (OSDictionary* dict, void* arg);

class AppleSmartBattery : public IOPMPowerSource {
    OSDeclareDefaultStructors(AppleSmartBattery)
protected:
    AppleSmartBatteryManager    *fProvider;
    IOWorkLoop                  *fWorkLoop;
    IOTimerEventSource          *fBatteryReadAllTimer;
    bool                        fInflowDisabled;
    bool                        fChargeInhibited;
    bool                        fCapacityOverride;
    bool                        fMaxCapacityOverride;

    uint8_t                     fInitialPollCountdown;

    IOService *                 fPowerServiceToAck;
    bool                        fSystemSleeping;

    bool                        fPermanentFailure;
    bool                        fFullyDischarged;
    bool                        fFullyCharged;
    bool                        fBatteryPresent;
    int                         fACConnected;
    int                         fInstantCurrent;
    OSArray                     *fCellVoltages;
    uint64_t                    acAttach_ts;

    CommandTable                cmdTable;
    bool                        fDisplayKeys;
    OSSet                       *fReportersSet;
    // Wrapper around IOPMPowerSource::setExternalConnected()
    void    setExternalConnectedToIOPMPowerSource(bool);

    CommandStruct *commandForState(uint32_t state);
    void    initializeCommands(void);
    bool    initiateTransaction(CommandStruct *cs);
    bool    doInitiateTransaction(const CommandStruct *cs);
    bool    initiateNextTransaction(uint32_t state);
    bool    retryCurrentTransaction(uint32_t state);
    bool    handleSetItAndForgetIt(int state, int val16,
                                   const uint8_t *str32, IOByteCount len);
    void    readAdapterInfo(void);
    OSDictionary* copySMCAdapterInfo(uint8_t port);
    IOReturn setChargeRateLimit(OSObject *value);
    IOReturn setChargeLimitDisplay(bool enable);
    void checkBatteryId(void);

public:
    static AppleSmartBattery *smartBattery(void);
    virtual bool init(void) APPLE_KEXT_OVERRIDE;
    virtual bool start(IOService *provider) APPLE_KEXT_OVERRIDE;
    bool    pollBatteryState(int path);
    void    handleBatteryInserted(void);
    void    handleBatteryRemoved(void);
    void    handleChargeInhibited(bool charge_state);
    void    handleExclusiveAccess(bool exclusive);
    void    handleSetOverrideCapacity(uint16_t value);
    void    handleSwitchToTrueCapacity(void);
    IOReturn handleSystemSleepWake(IOService * powerService, bool isSystemSleep);
    IOReturn newUserClient(task_t owningTask, void *securityID, UInt32 type, IOUserClient **handler) APPLE_KEXT_OVERRIDE;
    

protected:
    void    clearBatteryState(bool do_update);

    void    incompleteReadTimeOut(void);

    void    rebuildLegacyIOBatteryInfo(void);

    void    transactionCompletion(void *ref, IOReturn status, IOByteCount inCount, uint8_t *inData);

    void    handlePollingFinished(bool visitedEntirePath);

    void    acknowledgeSystemSleepWake( void );
    IOReturn createReporters(void);
    IOReturn updateChannelValue(IOSimpleReporter * reporter, uint64_t channel, OSObject *obj);
    IOReturn updateChannelValue(IOSimpleReporter * reporter, uint64_t channel, SInt64 value);
    IOReturn configureReport(IOReportChannelList *channels, IOReportConfigureAction action,
                                   void *result, void *destination) APPLE_KEXT_OVERRIDE;
    IOReturn updateReport(IOReportChannelList *channels, IOReportUpdateAction action,
                                void *result, void *destination) APPLE_KEXT_OVERRIDE;
#if TARGET_OS_IPHONE || TARGET_OS_OSX_AS
    void         checkBatteryIdNotification(void *param, IOService *charger, IONotifier *notifier);
    void         externalConnectedTimeout(void);
    uint32_t    _gasGaugeFirmwareVersion;
#endif // TARGET_OS_IPHONE || TARGET_OS_OSX_AS
    
private:
    // poll loop control use lock to access
    IORWLock *_pollCtrlLock;
    bool _cancelPolling;
    bool _pollingNow;
    uint16_t _machinePath;
    bool _rebootPolling;
    // -------------

    size_t _batteryCellCount;

    IOReturn setPropertiesGated(OSDictionary *dict);
    IOReturn handleSystemSleepWakeGated(IOService * powerService, bool isSystemSleep);
    void acknowledgeSystemSleepWakeGated(void);
    void handleBatteryRemovedGated(void);
    void updateDictionaryInIOReg(const OSSymbol *sym, smcToRegistry *keys);
    IOReturn transactionCompletionGated(struct transactionCompletionGatedArgs *args);
    bool initiateTransactionGated(CommandStruct *cs);
    void clearBatteryStateGated(bool do_update);
    void rebuildLegacyIOBatteryInfoGated(void);
    void handlePollingFinishedGated(bool visitedEntirePath, uint16_t machinePath);
    void handleSetOverrideCapacityGated(uint16_t value, bool sticky);
    void handleSwitchToTrueCapacityGated(void);
};

#endif
