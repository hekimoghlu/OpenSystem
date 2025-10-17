/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
//  main.m
//  PythonWrapper Test App
//
//  Created by Andy Kaplan on 3/29/21.
//

#import <Cocoa/Cocoa.h>
#import <Python/Python.h>

#define TEST_EXEC 0
#define TEST_NSTASK 0
#define TEST_FRAMEWORK 0
#define TEST_MULTITHREADING 1

#define PBINPATH "/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python"

#if TEST_EXEC | TEST_MULTITHREADING
static int testExec(void) {
    NSLog(@"Testing python execv prompt");
    static char * args[] = {PBINPATH, "-c", "with open(\"/private/tmp/ptest.log\", \"a\") as fp:\n\tfp.write(\"hi\\n\")", NULL};
    if (fork() == 0) {
        if (execv(PBINPATH, args) == -1) {
            NSLog(@"Error executing python: %s\n", strerror(errno));
        }
        return 1;
    }
    return 0;
}
#endif // TEST_EXEC | TEST_MULTITHREADING

#if TEST_NSTASK
static int testNSTask(void) {
    NSLog(@"Testing python NSTask prompt");
    if (fork() == 0) {
        NSTask *task = [[NSTask alloc] init];

        task.launchPath = @PBINPATH;
        task.arguments = [NSArray arrayWithObjects: @"-c", @"with open(\"/private/tmp/ptest2.log\", \"a\") as fp:\n\tfp.write(\"hi2\\n\")", nil];

        task.standardInput = [NSPipe pipe];
        task.standardOutput = [NSPipe pipe];
        task.standardError = [NSPipe pipe];
        [task launch];
        [task waitUntilExit];

        NSInteger exitCode = task.terminationStatus;

        NSData *outputData = [[task.standardOutput fileHandleForReading] readDataToEndOfFile];
        NSString *resultString = [[NSString alloc] initWithData:outputData encoding:NSUTF8StringEncoding];
        if (resultString.length > 0) {
            printf("%s", resultString.UTF8String);
        }
        return (int)exitCode;
    }
    return 0;
}
#endif // TEST_NSTASK
        
#if TEST_FRAMEWORK
static int testFramework(void) {
    NSLog(@"Testing python framework prompt");
    Py_Initialize();
    // The script should keep printing 5 statements, even while the prompt is displayed
    for (int i = 1; i <= 5; i++) {
        NSLog(@"Testing python framework prompt %d", i);
        sleep(1);
    }
    return 0;
}
#endif // TEST_FRAMEWORK

int main(int argc, const char * argv[]) {
    int rv  = 0;
    
    @autoreleasepool {
    
#if TEST_EXEC | TEST_MULTITHREADING
#if !TEST_MULTITHREADING
        int threads = 1;
#else // TEST_MULTITHREADING
        // Expect to see ~"threads" number of prompts if file locking is not functioning, or only 1 if it is.
        int threads = 5;
#endif // !TEST_MULTITHREADING
        for (int i = 0; i < threads; i++) {
            rv = testExec();
            if (rv) {
                break;
            }
        }
#endif // TEST_EXEC | TEST_MULTITHREADING

#if TEST_NSTASK
        rv = testNSTask();
#endif // TEST_NSTASK
        
#if TEST_FRAMEWORK
        rv = testFramework();
#endif // TEST_FRAMEWORK
        
        if (rv) {
            NSLog(@"Error executing test: %d", rv);
        }
    }
    return NSApplicationMain(argc, argv);
}
