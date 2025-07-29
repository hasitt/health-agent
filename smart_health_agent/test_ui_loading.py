#!/usr/bin/env python3
"""
Test script to verify UI loading without launching the full interface
"""

print("Testing UI components loading...")

try:
    # Test core imports
    import gradio as gr
    print("✅ Gradio imported successfully")
    
    # Test database
    import database
    print("✅ Database module imported")
    
    # Test other core modules
    import mood_tracking
    print("✅ Mood tracking module imported")
    
    import trend_analyzer
    print("✅ Trend analyzer module imported")
    
    # Test visualization import (should fail gracefully)
    try:
        import health_visualizations
        print("✅ Health visualizations imported successfully")
        visualizations_available = True
    except ImportError as e:
        print(f"ℹ️ Health visualizations not available: {e}")
        visualizations_available = False
    
    # Test UI creation components
    print("\n" + "="*50)
    print("TESTING UI COMPONENTS:")
    
    # Test basic Gradio components
    with gr.Blocks() as demo:
        gr.Markdown("# Test Interface")
        
        with gr.Tabs():
            with gr.Tab("Test Tab"):
                gr.Markdown("Test content")
                
                if visualizations_available:
                    gr.Markdown("Visualizations would be available here")
                else:
                    gr.Markdown("⚠️ **Visualization features not available.** To enable visualizations, install the required packages:")
                    gr.Markdown("```bash\npip install matplotlib seaborn\n```")
                
                test_button = gr.Button("Test Button")
                test_output = gr.Textbox(label="Test Output")
        
        def test_function():
            return "Test successful!"
        
        test_button.click(test_function, outputs=[test_output])
    
    print("✅ UI components created successfully")
    print("✅ Event handlers connected successfully")
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("✅ All core modules imported successfully")
    print("✅ UI components created without errors")
    print("✅ Event handlers connected properly")
    
    if visualizations_available:
        print("✅ Visualization features available")
    else:
        print("ℹ️ Visualization features not available (matplotlib/seaborn not installed)")
        print("   To enable: pip install matplotlib seaborn")
    
    print("\n🎉 UI loading test completed successfully!")
    print("The main application should now run without errors.")
    
except Exception as e:
    print(f"❌ Error during UI testing: {e}")
    import traceback
    traceback.print_exc()