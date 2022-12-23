#import <Cocoa/Cocoa.h>
#import "txt2img.h"

@interface IntegerFormatter : NSNumberFormatter
@end

@interface PromptFormatter : NSFormatter
@end

@interface Label : NSTextField
- (instancetype)initWithFrame:(NSRect)frame;
- (void)styleAsTitle;
@end

@interface RightPanel : NSView {
  NSTextField *promptBox;
  NSProgressIndicator *runProgress;
  Label* runStatus;
  NSButton *runButton;
  NSButton *saveButton;

  NSTextField *strengthBox;
  NSTextField *stepsBox;
  NSTextField *guideBox;
  NSTextField *seedBox;

  NSStepper *strengthStep;
  NSStepper *stepsStep;
  NSStepper *guideStep;
  NSStepper *seedStep;
}
- (instancetype)initWithFrame:(NSRect)frame;
- (void)fillRequest:(t2i_request_t*)req;
- (void)setRunStatus:(NSString*)string progress:(double)progress;
- (void)setRunEnabled:(BOOL)enabled;
- (IBAction)setStrength:(id)sender;
- (IBAction)setSteps:(id)sender;
- (IBAction)setGuide:(id)sender;
- (IBAction)setSeed:(id)sender;
@end

@interface Window : NSWindow {
  NSSplitView *splitView;
  NSImageView *imageView;
  RightPanel *rightPanel;
  t2i_t engine;
  int submit_id;
}
- (instancetype)init;
- (int)onStatus:(int)status req:(int)req_id;
- (BOOL)windowShouldClose:(id)sender;
- (IBAction)runModel:(id)sender;
- (IBAction)saveImage:(id)sender;
@end

@implementation IntegerFormatter
- (BOOL)isPartialStringValid:(NSString*)partialString
    newEditingString:(NSString**)newString errorDescription:(NSString**)error {
  if ([partialString length] == 0) return YES;
  NSScanner* scanner = [NSScanner scannerWithString:partialString];
  int value; BOOL allInteger = [scanner scanInt:&value] && [scanner isAtEnd];
  return allInteger && value >= 0 && value <= 100;
}
@end

@implementation PromptFormatter
- (NSString*)stringForObjectValue:(id)object {
  return (NSString*)object;
}

- (BOOL)getObjectValue:(id*)object forString:(NSString*)string errorDescription:(NSString**)error {
  *object = string;
  return YES;
}

- (BOOL)isPartialStringValid:(NSString**)partialStringPtr
    proposedSelectedRange:(NSRangePointer)newRange originalString:(NSString*)oldString
    originalSelectedRange:(NSRange)oldRange errorDescription:(NSString**)error {
  return [*partialStringPtr length] < N_TOKENS;
}
@end

@implementation Label
- (instancetype)initWithFrame:(NSRect)frame {
  self = [super initWithFrame:frame];
  if (self == nil) return self;

  super.bezeled = NO;
  super.drawsBackground = NO;
  super.editable = NO;
  super.selectable = NO;
  return self;
}

- (void)styleAsTitle {
  [super setFont:[NSFont boldSystemFontOfSize:NSFont.systemFontSize]];
  [super setTextColor:NSColor.secondaryLabelColor];
}
@end

@implementation RightPanel
- (void)addIntegerOption:(NSString*)name value:(int)value action:(SEL)action
    text:(NSTextField**)text stepper:(NSStepper**)stepper {
  Label *label = [[[Label alloc] initWithFrame:CGRectZero] autorelease];
  [label setTranslatesAutoresizingMaskIntoConstraints:NO];
  [label setStringValue:name];
  [self addSubview:label];

  NSTextField *option = [[[NSTextField alloc] initWithFrame:CGRectZero] autorelease];
  [option setTranslatesAutoresizingMaskIntoConstraints:NO];
  IntegerFormatter *formatter = [[[IntegerFormatter alloc] init] autorelease];
  [option setFormatter:formatter];
  [option setIntegerValue:value];
  [option setTarget:self];
  [option setAction:action];
  [self addSubview:option];

  NSStepper *control = [[[NSStepper alloc] initWithFrame:CGRectZero] autorelease];
  [control setTranslatesAutoresizingMaskIntoConstraints:NO];
  [control setMaxValue:100];
  [control setIntegerValue:value];
  [control setAction:action];
  [self addSubview:control];

  [self addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"H:|-[label(==104)]-[option]-[control]-|"
    options:NSLayoutFormatAlignAllCenterY metrics:nil
    views:NSDictionaryOfVariableBindings(label, option, control)]];
  *text = option, *stepper = control;
}

- (instancetype)initWithFrame:(NSRect)frame {
  self = [super initWithFrame:frame];
  if (self == nil) return self; 

  /* Create prompt control group */
  Label *promptTitle = [[[Label alloc] initWithFrame:CGRectZero] autorelease];
  [promptTitle setTranslatesAutoresizingMaskIntoConstraints:NO];
  [promptTitle setStringValue:@"Prompt"];
  [promptTitle styleAsTitle];
  [self addSubview:promptTitle];

  promptBox = [[NSTextField alloc] initWithFrame:CGRectZero];
  [promptBox setTranslatesAutoresizingMaskIntoConstraints:NO];
  PromptFormatter *formatter = [[[PromptFormatter alloc] init] autorelease];
  [promptBox setFormatter:formatter];
  [promptBox setStringValue:@"discovering ancient ruins, concept art by JaeCheol Park"];
  [self addSubview:promptBox];

  runProgress = [[NSProgressIndicator alloc] initWithFrame:CGRectZero];
  [runProgress setTranslatesAutoresizingMaskIntoConstraints:NO];
  [runProgress setStyle:NSProgressIndicatorStyleSpinning];
  [runProgress setControlSize:NSControlSizeSmall];
  [runProgress setIndeterminate:NO];
  [runProgress setDoubleValue:0];
  [self addSubview:runProgress];

  runStatus = [[Label alloc] initWithFrame:CGRectZero];
  [runStatus setTranslatesAutoresizingMaskIntoConstraints:NO];
  [runStatus setStringValue:@"Initializing"];
  [runStatus setTextColor:NSColor.secondaryLabelColor];
  [self addSubview:runStatus];

  runButton = [[NSButton alloc] initWithFrame:CGRectZero];
  [runButton setTranslatesAutoresizingMaskIntoConstraints:NO];
  [runButton setTitle:@"Run"];
  [runButton setBezelStyle:NSBezelStyleRounded];
  [runButton setEnabled:NO];
  [runButton setAction:@selector(runModel:)];
  [self addSubview:runButton];

  /* Constrain layout of prompt controls */
  [self addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"V:|-[promptTitle]-16-[promptBox(==80)]-16-[runProgress(==16)]"
    options:NSLayoutFormatAlignAllLeft metrics:nil
    views:NSDictionaryOfVariableBindings(promptTitle, promptBox, runProgress)]];
  [self addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"H:|-[promptBox(>=232)]-|"
    options:NSLayoutFormatAlignAllTop metrics:nil
    views:NSDictionaryOfVariableBindings(promptBox)]];
  [self addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"H:|-[runProgress(==16)]-[runStatus]-(>=16)-[runButton]-|"
    options:NSLayoutFormatAlignAllCenterY metrics:nil
    views:NSDictionaryOfVariableBindings(runProgress, runStatus, runButton)]];

  /* Create model options control group */
  NSBox *optionsLine = [[[NSBox alloc] initWithFrame:CGRectZero] autorelease];
  [optionsLine setTranslatesAutoresizingMaskIntoConstraints:NO];
  [optionsLine setBoxType:NSBoxSeparator];
  [self addSubview:optionsLine];

  Label *optionsTitle = [[[Label alloc] initWithFrame:CGRectZero] autorelease];
  [optionsTitle setTranslatesAutoresizingMaskIntoConstraints:NO];
  [optionsTitle setStringValue:@"Model Options"];
  [optionsTitle styleAsTitle];
  [self addSubview:optionsTitle];

  [self addIntegerOption:@"Strength" value:50
    action:@selector(setStrength:) text:&strengthBox stepper:&strengthStep];
  [self addIntegerOption:@"Denoising Steps" value:21
    action:@selector(setSteps:) text:&stepsBox stepper:&stepsStep];
  [self addIntegerOption:@"Guidance Scale" value:75
    action:@selector(setGuide:) text:&guideBox stepper:&guideStep];
  [self addIntegerOption:@"Random Seed" value:42
    action:@selector(setSeed:) text:&seedBox stepper:&seedStep];

  saveButton = [[NSButton alloc] initWithFrame:CGRectZero];
  [saveButton setTranslatesAutoresizingMaskIntoConstraints:NO];
  [saveButton setTitle:@"Save Image"];
  [saveButton setBezelStyle:NSBezelStyleRounded];
  [saveButton setAction:@selector(saveImage:)];
  [self addSubview:saveButton];

  /* Constrain layout of model options controls */
  [self addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"V:[runProgress]-32-[optionsLine]-[optionsTitle]"
    options:NSLayoutFormatAlignAllLeft metrics:nil
    views:NSDictionaryOfVariableBindings(runProgress, optionsLine, optionsTitle)]];
  [self addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"V:[optionsTitle]-16-[strengthBox]"
      "-[stepsBox]-[guideBox]-[seedBox]-16-[saveButton]-(>=16)-|"
    options:NSLayoutFormatDirectionLeftToRight metrics:nil
    views:NSDictionaryOfVariableBindings(optionsTitle, strengthBox, stepsBox, guideBox, seedBox, saveButton)]];
  [self addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"H:|-[optionsLine]-|"
    options:NSLayoutFormatAlignAllTop metrics:nil
    views:NSDictionaryOfVariableBindings(optionsLine)]];
  [self addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"H:[saveButton]-|"
    options:NSLayoutFormatAlignAllTop metrics:nil
    views:NSDictionaryOfVariableBindings(saveButton)]];

  return self;
}

- (void)fillRequest:(t2i_request_t*)req {
  NSString *prompt = [promptBox stringValue];
  NSCharacterSet *set = [NSCharacterSet whitespaceAndNewlineCharacterSet];
  prompt = [prompt lowercaseString];
  prompt = [prompt stringByTrimmingCharactersInSet:set];

  strncpy(req->prompt, [prompt UTF8String], N_TOKENS);
  req->strength = [strengthBox integerValue];
  req->steps = [stepsBox integerValue];
  req->guide = [guideBox integerValue];
  req->seed = [seedBox integerValue];
}

- (void)setRunStatus:(NSString*)string progress:(double)progress {
  [runStatus setStringValue:string];
  [runProgress setDoubleValue:progress];
}

- (void)setRunEnabled:(BOOL)enabled {
  [runButton setEnabled:enabled];
}

- (IBAction)setStrength:(id)sender {
  [strengthBox setIntegerValue:[sender integerValue]];
  [strengthStep setIntegerValue:[sender integerValue]];
}

- (IBAction)setSteps:(id)sender {
  [stepsBox setIntegerValue:[sender integerValue]];
  [stepsStep setIntegerValue:[sender integerValue]];
}

- (IBAction)setGuide:(id)sender {
  [guideBox setIntegerValue:[sender integerValue]];
  [guideStep setIntegerValue:[sender integerValue]];
}

- (IBAction)setSeed:(id)sender {
  [seedBox setIntegerValue:[sender integerValue]];
  [seedStep setIntegerValue:[sender integerValue]];
}
@end

static int handler(void *ctx, int req_id, int status) {
  __block int handler_err = 0;
  dispatch_async_and_wait(dispatch_get_main_queue(), ^{
    Window *window = (__bridge Window*)ctx;
    handler_err = [window onStatus:status req:req_id];
  });
  return handler_err;
}

@implementation Window
- (NSImage*)createImage:(char*)bytes {
  NSData *data = [NSData dataWithBytes:bytes length:512 * 512 * 3];
  CGColorSpaceRef colorspace = CGColorSpaceCreateDeviceRGB();
  CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
  CGImageRef imageRef = CGImageCreate(512, 512, 8, 8 * 3, 512 * 3, colorspace,
    kCGImageAlphaNone | kCGBitmapByteOrderDefault, provider, NULL, false, kCGRenderingIntentDefault);

  NSBitmapImageRep *bitmapRep = [[NSBitmapImageRep alloc] initWithCGImage:imageRef];
  NSImage *image = [[NSImage alloc] init];
  [image addRepresentation:bitmapRep];
  return image;
}

- (instancetype)init {
  /* create UI components */
  splitView = [[NSSplitView alloc] initWithFrame:CGRectZero];
  [splitView setTranslatesAutoresizingMaskIntoConstraints:NO];
  [splitView setDividerStyle:NSSplitViewDividerStylePaneSplitter];
  [splitView setVertical:YES];

  engine = t2i_init(handler, self);
  imageView = [[NSImageView alloc] initWithFrame:CGRectZero];
  [imageView setImage:[self createImage:t2i_request(engine, 0)->image]];
  [splitView addSubview:imageView];
  rightPanel = [[RightPanel alloc] initWithFrame:CGRectZero];
  [splitView addSubview:rightPanel];

  /* Setup window view */
  NSRect frame = NSMakeRect(100, 100, 720, 480);
  [super initWithContentRect:frame
    styleMask:NSWindowStyleMaskTitled
      | NSWindowStyleMaskClosable
      | NSWindowStyleMaskMiniaturizable
      | NSWindowStyleMaskResizable
    backing:NSBackingStoreBuffered
    defer:NO];
  [self setTitle:@"Diffusion.app"];

  /* Add subviews to window */
  [[self contentView] addSubview:splitView];
  [[self contentView] addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"H:|-0-[splitView]-0-|"
    options:NSLayoutFormatDirectionLeadingToTrailing metrics:nil
    views:NSDictionaryOfVariableBindings(splitView)]];
  [[self contentView] addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"V:|-0-[splitView]-0-|"
    options:NSLayoutFormatDirectionLeadingToTrailing metrics:nil
    views:NSDictionaryOfVariableBindings(splitView)]];
  [self setIsVisible:YES];
  return self;
}

- (int)onStatus:(int)status req:(int)req_id {
  NSLog(@"Received status %d for %d", status, req_id);
  if (req_id != -1 && req_id != submit_id) return 1;

  if (status == T2I_UNLOADED) {
    [rightPanel setRunStatus:@"Loading Encoder" progress:10];
  } else if (status == T2I_ENCODER_LOADED) {
    [rightPanel setRunStatus:@"Loading Diffuser" progress:30];
  } else if (status == T2I_ENCODER_NOLOAD) {
    [rightPanel setRunStatus:@"Encoder Invalid" progress:100];
  } else if (status == T2I_ENCODER_FAILED) {
    [rightPanel setRunStatus:@"Encoding Failed" progress:100];
  } else if (status == T2I_UNET_LOADED) {
    [rightPanel setRunStatus:@"Loading Decoder" progress:80];
  } else if (status == T2I_UNET_NOLOAD) {
    [rightPanel setRunStatus:@"Diffuser Invalid" progress:100];
  } else if (status == T2I_UNET_FAILED) {
    [rightPanel setRunStatus:@"Inference Failed" progress:100];
  } else if (status == T2I_DECODER_LOADED) {
    [rightPanel setRunStatus:@"Loading finished" progress:100];
    [rightPanel setRunEnabled:YES];
  } else if (status == T2I_DECODER_NOLOAD) {
    [rightPanel setRunStatus:@"Decoder Invalid" progress:100];
  } else if (status == T2I_DECODER_FAILED) {
    [rightPanel setRunStatus:@"Decoding Failed" progress:100];
  } else if (status >= T2I_STEPS) {
    int steps = status - T2I_STEPS, total = t2i_request(engine, req_id)->steps;
    NSString *str = [NSString stringWithFormat:@"Running step %d / %d", steps, total];
    [rightPanel setRunStatus:str progress:steps * 100 / total];
  } else if (status == T2I_FINISHED) {
    [rightPanel setRunStatus:@"Inference Finished" progress:100];
  }
  return 0;
}

- (BOOL)windowShouldClose:(id)sender {
  [NSApp terminate:sender];
  return YES;
}

- (IBAction)runModel:(id)sender {
  /* Attempt to acquire id */
  int req_id = t2i_acquire(engine);
  if (req_id == -1) return;

  /* Submit inference request */
  [rightPanel fillRequest:t2i_request(engine, req_id)];
  t2i_submit(engine, submit_id = req_id);
  [imageView setImage:[self createImage:t2i_request(engine, submit_id)->image]];
  [rightPanel setRunStatus:@"Starting Inference" progress:0];
}

- (IBAction)saveImage:(id)sender {
  NSSavePanel* panel = [NSSavePanel savePanel];
}
@end

int main(int argc, char *argv[]) {
  [NSApplication sharedApplication];
  [[[[Window alloc] init] autorelease] makeMainWindow];
  [NSApp run];
}
