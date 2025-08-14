# User Guide - Meera Insurance Document Assistant

## Welcome to Meera! 🤖

Meera is your intelligent insurance document assistant, designed to help you quickly find answers to questions about your insurance policies, coverage details, and related documents. This guide will help you get the most out of your interactions with Meera.

## Getting Started

### First Time Setup

1. **Access the Interface**
   - Open your web browser
   - Navigate to the provided URL (typically `http://127.0.0.1:7860`)
   - You'll see the Meera chat interface

2. **Configure Your Settings**
   - **System Prompt**: This defines Meera's personality and behavior
     - Default: "You are Meera, an assistant AI chatbot."
     - You can customize this to make Meera more specific to your needs
   - **Max Tokens**: Controls the length of Meera's responses
     - Range: 10-300 tokens
     - Default: 100 tokens
     - Higher values = longer, more detailed responses

3. **Start Chatting**
   - Type your question in the chat box
   - Press Enter or click the send button
   - Wait for Meera's response (typically 2-8 seconds)

## How to Ask Questions

### Best Practices

#### ✅ Good Questions
- **Specific**: "What is the deductible for the Gold plan?"
- **Clear**: "What does my policy cover for dental procedures?"
- **Focused**: "What are the exclusions for pre-existing conditions?"

#### ❌ Avoid These
- **Too vague**: "Tell me about insurance"
- **Too broad**: "What's in my policy?"
- **Multiple questions**: "What's the deductible, copay, and coverage limits?"

### Question Types and Examples

#### 1. Coverage Questions
```
"What does my health insurance cover?"
"What procedures are covered under my dental plan?"
"Does my policy cover prescription medications?"
"What's covered for mental health services?"
```

#### 2. Cost-Related Questions
```
"What is my deductible amount?"
"What's the copay for specialist visits?"
"How much do I pay for prescription drugs?"
"What are my out-of-pocket maximums?"
```

#### 3. Policy Details
```
"What are the policy exclusions?"
"What's the waiting period for certain procedures?"
"How do I file a claim?"
"What's the appeals process?"
```

#### 4. Plan Comparison
```
"What's the difference between the Gold and Silver plans?"
"Which plan has better prescription coverage?"
"Compare deductibles across all plans"
```

#### 5. Procedural Questions
```
"How do I add a dependent to my policy?"
"What's the process for changing plans?"
"How do I get a new insurance card?"
```

## Understanding Meera's Responses

### Response Structure
Meera's responses typically include:
1. **Direct Answer**: The main answer to your question
2. **Supporting Details**: Additional relevant information
3. **Source Reference**: Information about which documents were used

### Response Quality Indicators

#### ✅ High-Quality Response
- Contains specific details and numbers
- References specific policy sections
- Provides actionable information
- Includes relevant context

#### ⚠️ Limited Response
- Generic information only
- "Sorry, I don't know" message
- Vague or incomplete details

### When Meera Says "I Don't Know"
This can happen when:
- The information isn't in your uploaded documents
- Your question is too vague
- The documents don't contain the specific details you're asking about

**Solutions:**
- Rephrase your question more specifically
- Ask about related topics that might be covered
- Check if you have the right documents uploaded

## Advanced Features

### Conversation Memory
Meera remembers your conversation history, so you can:
- Ask follow-up questions: "What about the Silver plan?"
- Reference previous topics: "How does that compare to what you mentioned earlier?"
- Build on previous answers: "You mentioned the deductible is $2500, what does that mean?"

### Context-Aware Responses
Meera uses the most relevant parts of your documents to answer questions, ensuring:
- Accurate information based on your specific policies
- Relevant details from your actual coverage
- Personalized responses for your situation

### Adjustable Response Length
Use the "Max Tokens" slider to control response length:
- **Short responses (10-50 tokens)**: Quick facts and numbers
- **Medium responses (50-150 tokens)**: Balanced detail and brevity
- **Long responses (150-300 tokens)**: Comprehensive explanations with context

## Tips for Better Results

### 1. Be Specific
Instead of: "What's covered?"
Try: "What medical procedures are covered under my health insurance plan?"

### 2. Use Keywords
Include relevant terms like:
- Deductible, copay, premium
- Coverage, benefits, exclusions
- Policy, plan, insurance
- Specific procedure names

### 3. Ask One Question at a Time
Instead of: "What's the deductible, copay, and coverage for dental?"
Try: "What's the deductible for dental procedures?" (then follow up)

### 4. Use Follow-up Questions
- "Can you explain that in more detail?"
- "What does that mean in practical terms?"
- "How does this compare to other plans?"

### 5. Reference Specific Plans
- "In the Gold plan..."
- "For the HSA option..."
- "Under the Bronze coverage..."

## Common Use Cases

### Scenario 1: Understanding Your Coverage
```
You: "What does my health insurance cover?"
Meera: [Provides overview of covered services]

You: "What about prescription drugs?"
Meera: [Explains prescription coverage]

You: "Are there any limitations on that?"
Meera: [Details any restrictions or limits]
```

### Scenario 2: Cost Analysis
```
You: "What are all the costs I need to pay?"
Meera: [Lists deductibles, copays, premiums]

You: "What's my maximum out-of-pocket?"
Meera: [Provides out-of-pocket maximum]

You: "How does that work with the deductible?"
Meera: [Explains the relationship between costs]
```

### Scenario 3: Plan Comparison
```
You: "What's the difference between Gold and Silver plans?"
Meera: [Compares key features]

You: "Which has better prescription coverage?"
Meera: [Analyzes prescription benefits]

You: "What about the costs?"
Meera: [Compares pricing structure]
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Slow Response Times
**Problem**: Meera takes a long time to respond
**Solutions**:
- Check your internet connection
- Reduce the "Max Tokens" setting
- Try a simpler question first

#### 2. Unclear Responses
**Problem**: Meera's answers are confusing or incomplete
**Solutions**:
- Ask for clarification: "Can you explain that differently?"
- Break down complex questions into smaller parts
- Use more specific language

#### 3. Missing Information
**Problem**: Meera can't find information you know should be there
**Solutions**:
- Check if the relevant documents are uploaded
- Try different keywords or phrases
- Ask about related topics that might contain the information

#### 4. Technical Issues
**Problem**: Interface not loading or errors occurring
**Solutions**:
- Refresh the page
- Check if the service is running
- Contact your system administrator

### Getting Help

#### When to Contact Support
- Technical issues with the interface
- Persistent "I don't know" responses for important questions
- Missing documents that should be available
- Inconsistent or incorrect information

#### Information to Provide
- Your specific question
- Meera's response
- What you expected to hear
- Any error messages
- Your system configuration

## Best Practices Summary

### ✅ Do This
- Ask specific, focused questions
- Use relevant keywords and terms
- Build on previous conversations
- Adjust response length as needed
- Ask follow-up questions for clarity

### ❌ Avoid This
- Vague or overly broad questions
- Multiple questions in one message
- Technical jargon unless necessary
- Expecting information not in your documents
- Rushing through complex topics

## Example Conversations

### Conversation 1: Understanding Deductibles
```
You: "What is a deductible?"
Meera: "A deductible is the amount you must pay for covered healthcare services before your insurance plan starts to pay. For example, if your deductible is $1,500, you pay the first $1,500 of covered services yourself."

You: "How does that work with my Gold plan?"
Meera: "In your Gold plan, you have a $1,000 deductible for individual coverage. This means you pay the first $1,000 of covered medical expenses each year before your insurance begins to pay its share."

You: "What happens after I meet the deductible?"
Meera: "After meeting your $1,000 deductible, you typically pay only copayments or coinsurance for covered services. For example, you might pay a $25 copay for office visits and 20% coinsurance for other services."
```

### Conversation 2: Coverage Comparison
```
You: "What's the difference between Gold and Bronze plans?"
Meera: "The Gold plan has a $1,000 deductible with 20% coinsurance, while the Bronze plan has a $6,000 deductible with 40% coinsurance. Gold plans generally have higher premiums but lower out-of-pocket costs when you need care."

You: "Which is better for someone who visits the doctor frequently?"
Meera: "The Gold plan would be better for frequent doctor visits because you'll reach the lower deductible faster and then pay less in copays and coinsurance. The Bronze plan is better for those who rarely need medical care."

You: "What about prescription coverage?"
Meera: "Both plans cover prescription drugs, but the Gold plan has lower copays: $10 for generic, $30 for brand-name, and $50 for specialty drugs. The Bronze plan has higher copays: $25 for generic, $50 for brand-name, and $100 for specialty drugs."
```

## Advanced Tips

### 1. Use Natural Language
Meera understands natural language, so you can ask questions the way you'd ask a human:
- "I'm confused about how deductibles work"
- "Can you help me understand my coverage?"
- "What should I know about filing claims?"

### 2. Leverage Context
Since Meera remembers your conversation, you can:
- Reference previous answers: "You mentioned the $1000 deductible..."
- Build on information: "Given what you said about coverage..."
- Compare options: "How does that compare to..."

### 3. Ask for Examples
When dealing with complex concepts:
- "Can you give me an example?"
- "What does that mean in practice?"
- "Show me how this works with a real scenario"

### 4. Use Progressive Disclosure
Start with broad questions, then get more specific:
1. "What does my policy cover?"
2. "What about dental coverage?"
3. "What are the limitations on dental procedures?"
4. "How much do I pay for fillings?"

## Conclusion

Meera is designed to make understanding your insurance documents easier and more accessible. By following these guidelines and using the tips provided, you'll be able to get the most accurate and helpful information from your insurance documents.

Remember:
- Be specific in your questions
- Use the conversation memory to build on previous answers
- Adjust response length based on your needs
- Don't hesitate to ask for clarification when needed

Happy chatting with Meera! 🎉
