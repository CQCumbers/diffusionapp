#import <Cocoa/Cocoa.h>

// design async api based on io_uring or ioctl?

typedef struct Request {
  char prompt[4096];
  int strength, steps;
  int guide, seed;
} request_t;

static request_t appControls;
static int appStatus;
static 

// Implement everything in mockup
// except image options/dropdown
// Connect to inference backend
@interface Label : NSTextField
- (instancetype)initWithFrame:(NSRect)frame;
- (void)styleAsTitle;
@end

@interface RightPanel : NSView {
  NSTextField *promptBox;
  NSProgressIndicator *runProgress;
  Label* runStatus;
  NSButton *runButton;

  NSTextField *strengthBox;
  NSTextField *stepsBox;
  NSTextField *guideBox;
  NSTextField *seedBox;
  NSButton *saveButton;
}
- (instancetype)initWithFrame:(NSRect)frame;
@end

@interface Window : NSWindow {
  NSSplitView *splitView;
  NSImageView *imageView;
  RightPanel *rightPanel;
}
- (instancetype)init;
- (BOOL)windowShouldClose:(id)sender;
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
- (NSTextField*)addNumberOption:(NSString*)name {
  Label *label = [[[Label alloc] initWithFrame:CGRectZero] autorelease];
  [label setTranslatesAutoresizingMaskIntoConstraints:NO];
  [label setStringValue:name];
  [self addSubview:label];

  NSTextField *option = [[[NSTextField alloc] initWithFrame:CGRectZero] autorelease];
  [option setTranslatesAutoresizingMaskIntoConstraints:NO];
  [option setStringValue:@"50"];
  [self addSubview:option];

  NSStepper *stepper = [[[NSStepper alloc] initWithFrame:CGRectZero] autorelease];
  [stepper setTranslatesAutoresizingMaskIntoConstraints:NO];
  [self addSubview:stepper];

  [self addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"H:|-[label(==104)]-[option]-[stepper]-|"
    options:NSLayoutFormatAlignAllCenterY metrics:nil
    views:NSDictionaryOfVariableBindings(label, option, stepper)]];
  return option;
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

  promptBox = [[[NSTextField alloc] initWithFrame:CGRectZero] autorelease];
  [promptBox setTranslatesAutoresizingMaskIntoConstraints:NO];
  [promptBox setStringValue:@"discovering ancient ruins, concept art by JaeCheol Park"];
  [self addSubview:promptBox];

  runProgress = [[[NSProgressIndicator alloc] initWithFrame:CGRectZero] autorelease];
  [runProgress setTranslatesAutoresizingMaskIntoConstraints:NO];
  [runProgress setStyle:NSProgressIndicatorStyleSpinning];
  [runProgress setDoubleValue:100];
  [self addSubview:runProgress];

  runStatus = [[[Label alloc] initWithFrame:CGRectZero] autorelease];
  [runStatus setTranslatesAutoresizingMaskIntoConstraints:NO];
  [runStatus setStringValue:@"Finished"];
  [runStatus setTextColor:NSColor.secondaryLabelColor];
  [self addSubview:runStatus];

  runButton = [[[NSButton alloc] initWithFrame:CGRectZero] autorelease];
  [runButton setTranslatesAutoresizingMaskIntoConstraints:NO];
  [runButton setTitle:@"Run"];
  [runButton setBezelStyle:NSBezelStyleRounded];
  [self addSubview:runButton];

  /* Constrain layout of prompt controls */
  [self addConstraints:[NSLayoutConstraint
    constraintsWithVisualFormat:@"V:|-[promptTitle]-16-[promptBox(>=40)]-16-[runProgress(==16)]"
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

  strengthBox = [self addNumberOption:@"Strength"];
  stepsBox = [self addNumberOption:@"Denoising Steps"];
  guideBox = [self addNumberOption:@"Guidance Scale"];
  seedBox = [self addNumberOption:@"Random Seed"];

  saveButton = [[[NSButton alloc] initWithFrame:CGRectZero] autorelease];
  [saveButton setTranslatesAutoresizingMaskIntoConstraints:NO];
  [saveButton setTitle:@"Save Image"];
  [saveButton setBezelStyle:NSBezelStyleRounded];
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
@end

@implementation Window
- (instancetype)init {
  /* create UI components */
  splitView = [[[NSSplitView alloc] initWithFrame:CGRectZero] autorelease];
  [splitView setTranslatesAutoresizingMaskIntoConstraints:NO];
  [splitView setDividerStyle:NSSplitViewDividerStylePaneSplitter];
  [splitView setVertical:YES];

  imageView = [[[NSImageView alloc] initWithFrame:CGRectZero] autorelease];
  [splitView addSubview:imageView];
  rightPanel = [[[RightPanel alloc] initWithFrame:CGRectZero] autorelease];
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

- (BOOL)windowShouldClose:(id)sender {
  [NSApp terminate:sender];
  return YES;
}
@end

int main(int argc, char *argv[]) {
  [NSApplication sharedApplication];
  [[[[Window alloc] init] autorelease] makeMainWindow];
  [NSApp run];
}
